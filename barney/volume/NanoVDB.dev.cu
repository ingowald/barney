// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// include barney.h first, so we know whether BARNEY_HAVE_NANOVDB is set 
#include "barney/barney.h"
#if BARNEY_HAVE_NANOVDB
#include "barney/volume/NanoVDB.h"
#include "barney/volume/MCAccelerator.h"
#include "rtcore/TraceInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {

  template<typename T>
  struct MCAccel_NanoVDB_Programs {
    static inline __rtc_device
    void bounds(const rtc::TraceInterface &ti,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    {
#if RTC_DEVICE_CODE
      MCVolumeAccel<NanoVDBDataSampler<T>>
        ::boundsProg(ti,geomData,bounds,primID);
#endif
    }
    
    static inline __rtc_device
    void intersect(rtc::TraceInterface &ti)
    {
#if RTC_DEVICE_CODE
      MCVolumeAccel<NanoVDBDataSampler<T>>
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

  
  template<typename T>
  struct MCIsoAccel_NanoVDB_Programs {
    static inline __rtc_device
    void bounds(const rtc::TraceInterface &ti,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    {
#if RTC_DEVICE_CODE
      MCIsoSurfaceAccel<NanoVDBDataSampler<T>>
        ::boundsProg(ti,geomData,bounds,primID);
#endif
    }
    
    static inline __rtc_device
    void intersect(rtc::TraceInterface &ti)
    {
#if RTC_DEVICE_CODE
      MCIsoSurfaceAccel<NanoVDBDataSampler<T>>
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
   
  using NanoVDBMC_float = MCVolumeAccel<NanoVDBDataSampler<float>>;
  typedef typename NanoVDBMC_float::DD bla;
  typedef MCAccel_NanoVDB_Programs<float> blubb;
  
  RTC_EXPORT_USER_GEOM(NanoVDBMC_float,
                       typename NanoVDBMC_float::DD,
                       MCAccel_NanoVDB_Programs<float>,false,false);
  RTC_EXPORT_USER_GEOM(NanoVDBMC_Iso_float,
                       typename MCIsoSurfaceAccel<NanoVDBDataSampler<float>>::DD,
                       MCIsoAccel_NanoVDB_Programs<float>,false,false);
}


#endif

