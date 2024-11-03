// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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
#include "barney/GlobalModel.h"
#include "barney/render/Renderer.h"
#include "barney/fb/FrameBuffer.h"
#include "barney/Context.h"

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
                      int *dR_count,
                      Ray *rayQueue,
                      TileDesc *tileDescs);
  
  void  DeviceContext::generateRays_sync()
  {
    SetActiveGPU forDuration(device);

    this->launch_sync();
    rays.swap();
    // rays.numActive = *rays.d_nextWritePos;
    // *rays.d_nextWritePos = 0;
    rays.numActive = rays.readNumActive();
    rays.resetWriteQueue();
  }
  
  void DeviceContext::shadeRays_sync()
  {
    SetActiveGPU forDuration(device);
    launch_sync();
    rays.swap();
    rays.numActive = rays.readNumActive();
    rays.resetWriteQueue();
  }

  void DeviceContext::traceRays_launch(GlobalModel *model)
  {
    DevGroup *dg = device->devGroup;
    Context *context = model->context;
    ModelSlot *modelSlot = model->getSlot(dg->lmsIdx);
    Context::PerSlot *contextSlot = context->getSlot(dg->lmsIdx);
    owlParamsSetPointer(dg->lp,"rays",rays.traceAndShadeReadQueue);
    owlParamsSet1i(dg->lp,"numRays",rays.numActive);
    owlParamsSetGroup(dg->lp,"world",
                      modelSlot->instances.group);
    owlParamsSetBuffer(dg->lp,"materials",
                       contextSlot->materialRegistry->buffer);
    owlParamsSetBuffer(dg->lp,"samplers",
                       contextSlot->samplerRegistry->buffer);
                        
    int bs = 1024;
    int nb = divRoundUp(rays.numActive,bs);
    if (nb)
      owlAsyncLaunch2DOnDevice(dg->rg,bs,nb,device->owlID,dg->lp);
  }
}
