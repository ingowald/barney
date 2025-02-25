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

#include "barney/render/OptixGlobals.h"
#include "barney/Context.h"
#include "barney/GlobalModel.h"
#include "barney/ModelSlot.h"
#include "barney/render/SamplerRegistry.h"
#include "barney/render/MaterialRegistry.h"
#include "rtcore/ComputeInterface.h"

namespace BARNEY_NS {

  void Context::traceRaysLocally(GlobalModel *globalModel)
  {
    // ------------------------------------------------------------------
    // launch all in parallel ...
    // ------------------------------------------------------------------
    for (auto model : globalModel->modelSlots)
      for (auto device : *model->devices) {
        SetActiveGPU forDuration(device);
        render::OptixGlobals dd;
        auto ctx     = model->slotContext;
        dd.rays      = device->rayQueue->traceAndShadeReadQueue;
        dd.numRays   = device->rayQueue->numActive;
        dd.world     = model->getInstanceAccel(device);
        dd.materials = ctx->materialRegistry->getDD(device);
        dd.samplers  = ctx->samplerRegistry->getDD(device);
        dd.globalIndex = device->globalIndex;
        int bs = 1024;
        int nb = divRoundUp(dd.numRays,bs);
        device->traceRays->launch(vec2i(nb,bs),
                                  &dd);
      }
    
    // ------------------------------------------------------------------
    // ... and sync 'til all are done
    // ------------------------------------------------------------------
    syncCheckAll();
  }
}

 
