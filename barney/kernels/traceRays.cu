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
#include "barney/render/RayQueue.h"
#include "rtcore/ComputeInterface.h"

namespace BARNEY_NS {

  void Context::traceRaysLocally(GlobalModel *globalModel,
                                 uint32_t rngSeed,
                                 bool needHitIDs)
  {
    double t0 = getCurrentTime();
    
    // ------------------------------------------------------------------
    // launch all in parallel ...
    // ------------------------------------------------------------------
    for (auto model : globalModel->modelSlots) {
      for (auto device : *model->devices) {
        SetActiveGPU forDuration(device);
        render::OptixGlobals dd;
        auto ctx     = model->slotContext;
        dd.rays      = device->rayQueue->traceAndShadeReadQueue.rays;
        dd.hitIDs
          = needHitIDs
          ? device->rayQueue->traceAndShadeReadQueue.hitIDs
          : 0;
        dd.numRays   = device->rayQueue->numActive;
        dd.world     = model->world->getDD(device);//,rngSeed);
        dd.accel     = model->getInstanceAccel(device);

        if (FromEnv::get()->logQueues) {
          std::stringstream ss;
          ss << "#bn: ## ray queue kernel TRACE rays " << dd.rays << std::endl;
          ss << "#bn: ## ray queue kernel TRACE hit ids " << dd.hitIDs << " need = " << int(needHitIDs) << std::endl;
          std::cout << ss.str();
        }

        if (dd.numRays == 0 || dd.accel == 0) {
          /* iw - it's perfectly valid for an app to 'render' a model
             that's empty, so it's possible that dd.world is 0. Just
             skip calling the trace kernel, which may not like getting
             called with size 0 */
        } else {
          int bs = 1024;
          int nb = divRoundUp(dd.numRays,bs);
          if (nb)
            device->traceRays->launch(/* bs,nb intentionally inverted:
                                         always have 1024 in width: */
                                      vec2i(bs,nb),
                                      &dd);
        }
      }
    }

    // ------------------------------------------------------------------
    // ... and sync 'til all are done
    // ------------------------------------------------------------------
    syncCheckAll();
  }
  
} // ::BARNEY_NS

 
