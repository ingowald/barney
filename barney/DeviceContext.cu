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

  /* see generateRays.cu for implementation */
  __global__
  void g_generateRays(Camera camera,
                      int rngSeed,
                      vec2i fbSize,
                      int *d_count,
                      Ray *rayQueue,
                      TileDesc *tileDescs);
  
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

  /*! see shadeRays.cu for implementation */
  __global__
  void g_shadeRays(AccumTile *accumTiles,
                   Ray *readQueue,
                   int numRays,
                   Ray *writeQueue,
                   int *d_nextWritePos);
  
  void DeviceContext::shadeRays_launch(TiledFB *fb)
  {
    SetActiveGPU forDuration(device);
    int numRays = rays.numActive;
    int bs = 1024;
    int nb = divRoundUp(numRays,bs);

    // std::cout << "*shading* " << numRays << " on device " << fb->device->globalIndex << std::endl;
    if (nb)
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
    owlParamsSetPointer(dg->lp,"rays",rays.readQueue);
    owlParamsSet1i(dg->lp,"numRays",rays.numActive);
    OWLGroup world = model->getDG(dg->ldgID)->instances.group;
    owlParamsSetGroup(dg->lp,"world",world);
    int bs = 1024;
    int nb = divRoundUp(rays.numActive,bs);
    // std::cout << "tracing " << rays.numActive << " rays on dg " << dg->ldgID << std::endl;
    if (nb)
      owlAsyncLaunch2DOnDevice(dg->rg,bs,nb,device->owlID,dg->lp);
  }
}
