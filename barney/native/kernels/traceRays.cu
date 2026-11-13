// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/render/OptixGlobals.h"
#include "native/Context.h"
#include "native/GlobalModel.h"
#include "native/ModelSlot.h"
#include "native/render/SamplerRegistry.h"
#include "native/render/MaterialRegistry.h"
#include "native/render/RayQueue.h"
#include "native/FromEnv.h"
#include <chrono>

namespace BARNEY_NS {
  namespace native {

    void Context::traceRaysLocally(const std::vector<World *> &worlds,
                                   uint32_t rngSeed,
                                   bool needHitIDs)
    {
      // ------------------------------------------------------------------
      // launch all in parallel ...
      // ------------------------------------------------------------------
      OptixGlobals dd;

      for (auto device : devices) {
        SetActiveGPU forDuration(device);
        dd.rays      = device->rayQueue->traceAndShadeReadQueue.rays;
        dd.hitIDs
          = needHitIDs
          ? device->rayQueue->traceAndShadeReadQueue.hitIDs
          : 0;
        dd.numRays   = device->rayQueue->numActive;
        World *world = worlds[device->deviceIdx];
        dd.world     = world->getDD();//,rngSeed);
        dd.accel     = world->getInstanceAccel();
        dd.cutPlane  = activeCutPlane;
        
        if (dd.numRays == 0 || dd.accel == 0) {
          /* iw - it's perfectly valid for an app to 'render' a model
             that's empty, so it's possible that dd.world is 0. Just
             skip calling the trace kernel, which may not like getting
             called with size 0 */
        } else {
          int bs = 256;
          int nb = divRoundUp(dd.numRays,bs);
          if (nb)
            device->traceRays->launch(/* bs,nb intentionally inverted:
                                         always have 1024 in width: */
                                      vec2i(bs,nb),
                                      &dd);
        }
      }
      
      // ------------------------------------------------------------------
      // ... and sync 'til all are done
      // ------------------------------------------------------------------
      for (auto device : devices) {
        SetActiveGPU forDuration(device);
        device->rtc->sync();
      }
    }

  }
} // ::BARNEY_NS

 
