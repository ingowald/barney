// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "barney/DeviceContext.h"
#include "barney/Ray.h"
#include "barney/Model.h"

namespace barney {
  
  DeviceContext::DeviceContext(Device::SP device)
    : device(device),
      rays(device.get())
  {}

  __global__
  void g_generateRays(Camera camera,
                      int rngSeed,
                      vec2i fbSize,
                      int *d_count,
                      Ray *rayQueue,
                      TileDesc *tileDescs)
  {
    __shared__ int l_count;
    if (threadIdx.x == 0)
      l_count = 0;

    // ------------------------------------------------------------------
    __syncthreads();
    
    int tileID = blockIdx.x;
    
    vec2i tileOffset = tileDescs[tileID].lower;
    int ix = (threadIdx.x % tileSize) + tileOffset.x;
    int iy = (threadIdx.x / tileSize) + tileOffset.y;

    Ray ray;
    ray.origin  = camera.lens_00;
    ray.direction
      = camera.dir_00
      + (ix+.5f)*camera.dir_du
      + (iy+.5f)*camera.dir_dv;
    ray.tMax = 1e30f;
    ray.instID  = -1;
    ray.geomID  = -1;
    ray.primID  = -1;
    ray.u       = 0.f;
    ray.v       = 0.f;
    ray.seed    = rngSeed;
    ray.pixelID = tileID * (tileSize*tileSize) + threadIdx.x;
    ray.hadHit = false;

    int pos = -1;
    if (ix < fbSize.x && iy < fbSize.y) 
      pos = atomicAdd(&l_count,1);

    // ------------------------------------------------------------------
    __syncthreads();
    if (threadIdx.x == 0) 
      l_count = atomicAdd(d_count,l_count);
    
    // ------------------------------------------------------------------
    __syncthreads();
    if (pos >= 0) 
      rayQueue[l_count + pos] = ray;
  }
  
  void  DeviceContext::generateRays_launch(TiledFB *fb,
                                         const Camera &camera,
                                         int rngSeed)
  {
    SetActiveGPU forDuration(fb->device);
    
    g_generateRays
      <<<fb->numActiveTiles,pixelsPerTile,0,device->launchStream>>>
      (camera,
       rngSeed,
       fb->numPixels,
       rays.d_nextWritePos,
       rays.writeQueue,
       fb->tileDescs);
  }
  
  void  DeviceContext::generateRays_sync()
  {
    SetActiveGPU forDuration(device);
    
    this->launch_sync();
    std::swap(rays.readQueue, rays.writeQueue);
    rays.numActive = *rays.d_nextWritePos;
    *rays.d_nextWritePos = 0;
  }
  
  __global__
  void g_shadeRays(AccumTile *accumTiles,
                   Ray *readQueue,
                   int numRays,
                   Ray *writeQueue,
                   int *d_nextWritePos)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= numRays) return;

    Ray ray = readQueue[tid];
    vec3f color = abs(normalize(ray.direction));
    if (ray.hadHit)
      color = vec3f(1.f);
    int tileID  = ray.pixelID / pixelsPerTile;
    int tileOfs = ray.pixelID % pixelsPerTile;
    accumTiles[tileID].accum[tileOfs] = vec4f(color,0.f);
  }
  
  void DeviceContext::shadeRays_launch(TiledFB *fb)
  {
    SetActiveGPU forDuration(device);
    int numRays = rays.numActive;
    int bs = 1024;
    int nb = divRoundUp(numRays,bs);
    
    PING; PRINT(numRays);
    g_shadeRays<<<nb,bs,0,device->launchStream>>>
      (fb->accumTiles,rays.readQueue,numRays,rays.writeQueue,rays.d_nextWritePos);
  }

  void DeviceContext::shadeRays_sync()
  {
    SetActiveGPU forDuration(device);
    launch_sync();
    std::swap(rays.readQueue, rays.writeQueue);
    rays.numActive = *rays.d_nextWritePos;
    *rays.d_nextWritePos = 0;
  }

  void DeviceContext::traceRays_launch(Model *model)
  {
    DevGroup *dg = device->devGroup;
    PING;
    PRINT(rays.numActive);
    owlParamsSetPointer(dg->lp,"rays",rays.readQueue);
    owlParamsSet1i(dg->lp,"numRays",rays.numActive);
    OWLGroup world = model->getDG(dg->ldgID)->instances.group;
    owlParamsSetGroup(dg->lp,"world",world);
    int bs = 1024;
    int nb = divRoundUp(rays.numActive,bs);
    owlAsyncLaunch2DOnDevice(dg->rg,bs,nb,device->owlID,dg->lp);
  }
}
