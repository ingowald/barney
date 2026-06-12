// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/render/OptixGlobals.h"
#include "barney/Context.h"
#include "barney/GlobalModel.h"
#include "barney/ModelSlot.h"
#include "barney/render/SamplerRegistry.h"
#include "barney/render/MaterialRegistry.h"
#include "rtcore/TraceInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {
  namespace render {

    struct TraceRays {
#if RTC_DEVICE_CODE
      inline __rtc_device static 
      void run(rtc::TraceInterface &ti);
#endif
    };

#if RTC_DEVICE_CODE
    inline __rtc_device 
    void TraceRays::run(rtc::TraceInterface &ti)
    {
      const int rayID
        = ti.getLaunchIndex().x
        + ti.getLaunchDims().x
        * ti.getLaunchIndex().y;

      auto &lp = OptixGlobals::get(ti);

      if (rayID >= lp.numRays)
        return;
      
      Ray &ray = lp.rays[rayID];
      
      vec3f dir = ray.dir;
      if (dir.x == 0.f) dir.x = 1e-6f;
      if (dir.y == 0.f) dir.y = 1e-6f;
      if (dir.z == 0.f) dir.z = 1e-6f;

      ray.hip_dbg = int(ray.dbg());
      // if (rayID > 100) return; // works
      // if (rayID > 200) return; // workd
      // if (rayID > 300) return;
      // if (rayID > 400) return;
      // if (rayID > 1000) return;
      // if (rayID > 2000) return;
      // if (rayID > 4000) return;
      if (ray.hip_dbg) printf("traceRays.dev.cu $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
      // if (!ray.hip_dbg) return;
      ti.traceRay(lp.accel,
                  ray.org,
                  dir,
                  0.f,
                  ray.tMax,
                  /* PRD */
                  (void *)&ray);
    }
#endif
    
  }
  
  RTC_EXPORT_TRACE2D(traceRays,render::TraceRays);
}

