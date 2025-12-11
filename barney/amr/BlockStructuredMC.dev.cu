// SPDX-FileCopyrightText:
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier:
// Apache-2.0

/*! \file BlockStructuredMC.dev.cu implements a macro-cell accelerated
    unstructured mesh data type.

    This particular voluem type:

    - uses cubql to accelerate point-in-element queries (for the
      scalar field evaluation)

    - uses macro cells and DDA traversal for domain traversal
*/

#include "barney/amr/BlockStructuredCuBQLSampler.h"
#include "barney/volume/DDA.h"
#include "rtcore/TraceInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {

  struct BlockStructuredMC_Programs {
    
    static inline __rtc_device
    void bounds(const rtc::TraceInterface &ti,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    {
#if RTC_DEVICE_CODE
      MCVolumeAccel<BlockStructuredCuBQLSampler>::boundsProg(ti,geomData,bounds,primID);
#endif
    }

    static inline __rtc_device
    void intersect(rtc::TraceInterface &ti)
    {
#if RTC_DEVICE_CODE
      MCVolumeAccel<BlockStructuredCuBQLSampler>::isProg(ti);
#endif
    }
    
    static inline __rtc_device
    void closestHit(rtc::TraceInterface &ti)
    { /* nothing to do */ }
    
    static inline __rtc_device
    void anyHit(rtc::TraceInterface &ti)
    { /* nothing to do */ }
  };

  using BlockStructuredMC = MCVolumeAccel<BlockStructuredCuBQLSampler>;

  RTC_EXPORT_USER_GEOM(BlockStructuredMC,
                       BlockStructuredMC::DD,
                       BlockStructuredMC_Programs,
                       false,false);
}

