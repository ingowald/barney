// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/umesh/mc/IconField.h"
#include "barney/volume/DDA.h"
#include "rtcore/TraceInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {

  struct IconField_Programs {
    
    static inline __rtc_device
    void anyHit(rtc::TraceInterface &ti)
    { /* ignore, not used, but has to exist */ }
    
    static inline __rtc_device
    void closestHit(rtc::TraceInterface &ti)
    { /* TODO: getPRD, then set appropriate triangle ID */
      auto &prd = *(IconMultiPassSampler::PRD *)ti.getPRD();
      prd.primID = ti.getPrimitiveIndex();
    }
    
  };


  struct IconField_TraceRays {
#if RTC_DEVICE_CODE
    inline __rtc_device static 
    void run(rtc::TraceInterface &ti);
#endif
  };

#if RTC_DEVICE_CODE
  inline __rtc_device 
  void IconField_TraceRays::run(rtc::TraceInterface &ti)
  {
    const int rayID
      = ti.getLaunchIndex().x
      + ti.getLaunchDims().x
      * ti.getLaunchIndex().y;
    if (rayID == 0)
      printf("iconfield whole-frame launch ...\n");
  }
#endif

  // using IconField = MCVolumeAccel<UMeshCuBQLSampler>;
  // using IconField_Iso = MCIsoSurfaceAccel<UMeshCuBQLSampler>;

  RTC_EXPORT_TRIANGLES_GEOM(IconField,IconMultiPassSampler::DD,
                            IconField_Programs,false,true);
  RTC_EXPORT_TRACE2D(traceRays_IconField,IconField_TraceRays);
}



