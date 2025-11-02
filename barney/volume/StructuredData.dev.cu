// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/volume/StructuredData.h"
#include "barney/volume/MCAccelerator.h"
#include "rtcore/TraceInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {

  struct MCAccel_Structured_Programs {
    static inline __rtc_device
    void bounds(const rtc::TraceInterface &ti,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    {
#if RTC_DEVICE_CODE
      MCVolumeAccel<StructuredDataSampler>
        ::boundsProg(ti,geomData,bounds,primID);
#endif
    }
    
    static inline __rtc_device
    void intersect(rtc::TraceInterface &ti)
    {
#if RTC_DEVICE_CODE
      MCVolumeAccel<StructuredDataSampler>
        ::isProg(ti);
#endif
    }
    
    static inline __rtc_device
    void closestHit(rtc::TraceInterface &ti)
    { /* nothing to do */ }
    
    static inline __rtc_device
    void anyHit(rtc::TraceInterface &ti)
    { /* nothing to do */ }
  };

  struct MCIsoAccel_Structured_Programs {
    static inline __rtc_device
    void bounds(const rtc::TraceInterface &ti,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    {
#if RTC_DEVICE_CODE
      MCIsoSurfaceAccel<StructuredDataSampler>
        ::boundsProg(ti,geomData,bounds,primID);
#endif
    }
    
    static inline __rtc_device
    void intersect(rtc::TraceInterface &ti)
    {
#if RTC_DEVICE_CODE
      MCIsoSurfaceAccel<StructuredDataSampler>
        ::isProg(ti);
#endif
    }
    
    static inline __rtc_device
    void closestHit(rtc::TraceInterface &ti)
    { /* nothing to do */ }
    
    static inline __rtc_device
    void anyHit(rtc::TraceInterface &ti)
    { /* nothing to do */ }
  };
  
  
  RTC_EXPORT_USER_GEOM(StructuredMC,
                       typename MCVolumeAccel<StructuredDataSampler>::DD,
                       MCAccel_Structured_Programs,false,false);
  RTC_EXPORT_USER_GEOM(StructuredMC_Iso,
                       typename MCIsoSurfaceAccel<StructuredDataSampler>::DD,
                       MCIsoAccel_Structured_Programs,false,false);
}



