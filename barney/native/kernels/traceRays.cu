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
    
    void Context::traceRaysLocally(GlobalModel *globalModel,
                                   uint32_t rngSeed,
                                   bool needHitIDs)
    {
      std::cout << "==================================================================" << std::endl;
      PING;
      // ------------------------------------------------------------------
      // launch all in parallel ...
      // ------------------------------------------------------------------
      OptixGlobals dd;
      int slotIdx = 0;

      typedef  std::chrono::time_point<std::chrono::high_resolution_clock> time_t;

      static time_t t_launch[2];
      static time_t t_launched[2];
      static time_t t_sync[2]; 
      static time_t t_synched[2];
      int which = 0;
      
      auto t0 = std::chrono::high_resolution_clock::now();;
      which = -1;
      for (auto model : globalModel->modelSlots) {
        for (auto device : *model->devices) {
          ++which; PRINT(which);
          SetActiveGPU forDuration(device);
          PRINT(device->toString());
          auto ctx     = model->ldgContext;
          dd.rays      = device->rayQueue->traceAndShadeReadQueue.rays;
          dd.hitIDs
            = needHitIDs
            ? device->rayQueue->traceAndShadeReadQueue.hitIDs
            : 0;
          dd.numRays   = device->rayQueue->numActive;
          dd.world     = model->world->getDD(device);//,rngSeed);
          dd.accel     = model->getInstanceAccel(device);
          dd.cutPlane  = activeCutPlane;

          if (FromEnv::logQueues) {
            std::stringstream ss;
            ss << "#bn(" << device->globalRank() << "): ## ray queue kernel TRACE rays " << dd.rays << std::endl;
            ss << "#bn(" << device->globalRank() << "): ## ray queue kernel TRACE hit ids " << dd.hitIDs << " need = " << int(needHitIDs) << std::endl;
            std::cout << ss.str();
          }

          PRINT(dd.numRays); PRINT(dd.accel);
          if (dd.numRays == 0 || dd.accel == 0) {
            /* iw - it's perfectly valid for an app to 'render' a model
               that's empty, so it's possible that dd.world is 0. Just
               skip calling the trace kernel, which may not like getting
               called with size 0 */
          } else {
            int bs = 64;
            int nb = divRoundUp(dd.numRays,bs);

            // if (myRank() == 0)
            //   printf(" -> tracing %i\n",dd.numRays);
            t_launch[which] = std::chrono::high_resolution_clock::now();

            PRINT(nb);
            if (nb)
              device->traceRays->launch(/* bs,nb intentionally inverted:
                                           always have 1024 in width: */
                                        vec2i(bs,nb),
                                        &dd);
            t_launched[which] = std::chrono::high_resolution_clock::now();
          }
        }
        slotIdx++;
      }

      // ------------------------------------------------------------------
      // ... and sync 'til all are done
      // ------------------------------------------------------------------
      which = -1;
      for (auto device : *devices) {
        ++which;
        t_sync[which] = std::chrono::high_resolution_clock::now();
        SetActiveGPU forDuration(device);
        device->rtc->sync();
        t_synched[which] = std::chrono::high_resolution_clock::now();
      }
      which = -1;
      if (FromEnv::logQueues) {
        std::stringstream ss;
        ss << "#bn(" << myRank() << "): ## ray queue kernel TRACE DONE" << std::endl;
        std::cout << ss.str();
      }

      PRINT(std::chrono::duration_cast<std::chrono::nanoseconds>(t_launch[0]-t0));
      PRINT(std::chrono::duration_cast<std::chrono::nanoseconds>(t_launch[1]-t0));
      PRINT(std::chrono::duration_cast<std::chrono::nanoseconds>(t_launched[0]-t0));
      PRINT(std::chrono::duration_cast<std::chrono::nanoseconds>(t_launched[1]-t0));
      PRINT(std::chrono::duration_cast<std::chrono::nanoseconds>(t_sync[0]-t0));
      PRINT(std::chrono::duration_cast<std::chrono::nanoseconds>(t_sync[1]-t0));
      PRINT(std::chrono::duration_cast<std::chrono::nanoseconds>(t_synched[0]-t0));
      PRINT(std::chrono::duration_cast<std::chrono::nanoseconds>(t_synched[1]-t0));
    }

  }
} // ::BARNEY_NS

 
