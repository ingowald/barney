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

#include "mori/MoriContext.h"

namespace mori {
  
  MoriContext::MoriContext(Device::SP device)
    : device(device),
      rays(device.get()),
      lp(createLP(device.get())),
      launchStream(owlParamsGetCudaStream(lp,0))
  {}


  OWLParams MoriContext::createLP(Device *device)
  {
    OWLVarDecl params[]
      = {
         { "world", OWL_GROUP, OWL_OFFSETOF(DD, world) },
         { "rayQueue", OWL_USER_TYPE(RayQueue::DD), OWL_OFFSETOF(DD,rayQueue) },
         { nullptr }
    };
    OWLParams lp = owlParamsCreate(device->owlContext,
                                   sizeof(DD),
                                   params,
                                   -1);
    return lp;
  }
    
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
  
  void  MoriContext::generateRays_launch(TiledFB *fb,
                                         const Camera &camera,
                                         int rngSeed)
  {
    SetActiveGPU(fb->device);
    g_generateRays
      <<<fb->numActiveTiles,pixelsPerTile,0,launchStream>>>
      (camera,
       rngSeed,
       fb->numPixels,
       rays.d_nextWritePos,
       rays.writeQueue,
       fb->tileDescs);
  }
  
  void  MoriContext::generateRays_sync()
  {
    this->launch_sync();
    rays.numActive = *rays.d_nextWritePos;
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
    int tileID  = ray.pixelID / pixelsPerTile;
    int tileOfs = ray.pixelID % pixelsPerTile;
    accumTiles[tileID].accum[tileOfs] = vec4f(color,0.f);
  }
  
  void MoriContext::shadeRays_launch(TiledFB *fb)
  {
    int numRays = *rays.d_nextWritePos;
    *rays.d_nextWritePos = 0;

    std::swap(rays.readQueue, rays.writeQueue);
    
    int bs = 1024;
    int nb = divRoundUp(numRays,bs);
    g_shadeRays<<<nb,bs,0,launchStream>>>
      (fb->accumTiles,rays.readQueue,numRays,rays.writeQueue,rays.d_nextWritePos);
  }
    
}
