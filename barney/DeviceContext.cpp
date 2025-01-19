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

  void  DeviceContext::generateRays_sync()
  {
    SetActiveGPU forDuration(device);

    device->rtc->sync();
    rays.swap();
    // rays.numActive = *rays.d_nextWritePos;
    // *rays.d_nextWritePos = 0;
    rays.numActive = rays.readNumActive();
    rays.resetWriteQueue();
  }
  
  void DeviceContext::shadeRays_sync()
  {
    SetActiveGPU forDuration(device);
    device->rtc->sync();
    rays.swap();
    rays.numActive = rays.readNumActive();
    rays.resetWriteQueue();
  }

  void DeviceContext::traceRays_launch(GlobalModel *model)
  {
    DevGroup *dg = device->devGroup;
    Context *context = model->context;
    ModelSlot *modelSlot = model->getSlot(dg->lmsIdx);
    const Context::PerSlot *contextSlot = context->getSlot(dg->lmsIdx);
    
    barney::render::OptixGlobals dd;
    dd.rays    = /* already a single device pointer */
      rays.traceAndShadeReadQueue;
    dd.numRays = rays.numActive;
    dd.world   = modelSlot->instances.group->getDD(device->rtc);
    dd.materials = contextSlot->materialRegistry->getDD(device->rtc);
    dd.samplers = contextSlot->samplerRegistry->getDD(device->rtc);
      
    //device->launchTrace(&dd);
    int bs = 1024;
    int nb = divRoundUp(rays.numActive,bs);
    getDevGroup()->traceRaysKernel->launch(device->rtc,vec2i(nb,bs),&dd);
  }
  
}
