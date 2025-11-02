// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


/*! \file BlockStructuredMC.dev.cu implements a macro-cell accelerated
    unstructured mesh data type.

    This particular voluem type:

    - uses cubql to accelerate point-in-element queries (for the
      scalar field evaluation)

    - uses macro cells and DDA traversal for domain traversal
*/

#include "barney/amr/BlockStructuredCUBQLSampler.h"
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
      MCVolumeAccel<BlockStructuredCUBQLSampler>::boundsProg(ti,geomData,bounds,primID);
#endif
    }

    static inline __rtc_device
    void intersect(rtc::TraceInterface &ti)
    {
#if RTC_DEVICE_CODE
      MCVolumeAccel<BlockStructuredCUBQLSampler>::isProg(ti);
#endif
    }
    
    static inline __rtc_device
    void closestHit(rtc::TraceInterface &ti)
    { /* nothing to do */ }
    
    static inline __rtc_device
    void anyHit(rtc::TraceInterface &ti)
    { /* nothing to do */ }
  };

  using BlockStructuredMC = MCVolumeAccel<BlockStructuredCUBQLSampler>;

  RTC_EXPORT_USER_GEOM(BlockStructuredMC,BlockStructuredMC::DD,BlockStructuredMC_Programs,false,false);
}



// // ======================================================================== //
// // Copyright 2023-2024 Ingo Wald                                            //
// //                                                                          //
// // Licensed under the Apache License, Version 2.0 (the "License");          //
// // you may not use this file except in compliance with the License.         //
// // You may obtain a copy of the License at                                  //
// //                                                                          //
// //     http://www.apache.org/licenses/LICENSE-2.0                           //
// //                                                                          //
// // Unless required by applicable law or agreed to in writing, software      //
// // distributed under the License is distributed on an "AS IS" BASIS,        //
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// // See the License for the specific language governing permissions and      //
// // limitations under the License.                                           //
// // ======================================================================== //

// #include "barney/amr/BlockStructuredCUBQLSampler.h"
// #include "rtcore/TraceInterface.h"

// RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

// namespace barney {

//   // ==================================================================
//   //
//   // Block-Structured Data, Macro-Cell (MC) accelerator, and traversal
//   // via an RTX BVH built over active macro-cells
//   //
//   // ================================================================== 

//   OPTIX_BOUNDS_PROGRAM(BlockStructured_MCRTX_Bounds)(const void *geomData,
//                                                 owl::common::box3f &bounds,
//                                                 const int32_t primID)
//   {
//     MCRTXVolumeAccel<BlockStructuredCUBQLSampler>::boundsProg
//       (geomData,bounds,primID);
//   }

//   OPTIX_INTERSECT_PROGRAM(BlockStructured_MCRTX_Isec)()
//   {
//     MCRTXVolumeAccel<BlockStructuredCUBQLSampler>::isProg();
//   }
  
//   OPTIX_CLOSEST_HIT_PROGRAM(BlockStructured_MCRTX_CH)()
//   {
//     MCRTXVolumeAccel<BlockStructuredCUBQLSampler>::chProg();
//   }
  



//   // ==================================================================
//   //
//   // Block-Structured Data, Macro-Cell (MC) accelerator, and cuda-DDA
//   // traversal
//   //
//   // ================================================================== 

//   OPTIX_BOUNDS_PROGRAM(BlockStructured_MCDDA_Bounds)(const void *geomData,
//                                                 owl::common::box3f &bounds,
//                                                 const int32_t primID)
//   {
//     MCDDAVolumeAccel<BlockStructuredCUBQLSampler>::boundsProg(geomData,bounds,primID);
//   }

//   OPTIX_INTERSECT_PROGRAM(BlockStructured_MCDDA_Isec)()
//   {
//     MCDDAVolumeAccel<BlockStructuredCUBQLSampler>::isProg();
//   }
  
//   OPTIX_CLOSEST_HIT_PROGRAM(BlockStructured_MCDDA_CH)()
//   {
//     MCDDAVolumeAccel<BlockStructuredCUBQLSampler>::chProg();
//   }

// }
