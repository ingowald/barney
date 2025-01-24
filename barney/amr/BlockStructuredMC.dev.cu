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

#include "barney/amr/BlockStructuredCUBQLSampler.h"

RTC_DECLARE_GLOBALS(barney::render::OptixGlobals);


#if 0
namespace barney {

  // ==================================================================
  //
  // Block-Structured Data, Macro-Cell (MC) accelerator, and traversal
  // via an RTX BVH built over active macro-cells
  //
  // ================================================================== 

  OPTIX_BOUNDS_PROGRAM(BlockStructured_MCRTX_Bounds)(const void *geomData,
                                                owl::common::box3f &bounds,
                                                const int32_t primID)
  {
    MCRTXVolumeAccel<BlockStructuredCUBQLSampler>::boundsProg
      (geomData,bounds,primID);
  }

  OPTIX_INTERSECT_PROGRAM(BlockStructured_MCRTX_Isec)()
  {
    MCRTXVolumeAccel<BlockStructuredCUBQLSampler>::isProg();
  }
  
  OPTIX_CLOSEST_HIT_PROGRAM(BlockStructured_MCRTX_CH)()
  {
    MCRTXVolumeAccel<BlockStructuredCUBQLSampler>::chProg();
  }
  



  // ==================================================================
  //
  // Block-Structured Data, Macro-Cell (MC) accelerator, and cuda-DDA
  // traversal
  //
  // ================================================================== 

  OPTIX_BOUNDS_PROGRAM(BlockStructured_MCDDA_Bounds)(const void *geomData,
                                                owl::common::box3f &bounds,
                                                const int32_t primID)
  {
    MCDDAVolumeAccel<BlockStructuredCUBQLSampler>::boundsProg(geomData,bounds,primID);
  }

  OPTIX_INTERSECT_PROGRAM(BlockStructured_MCDDA_Isec)()
  {
    MCDDAVolumeAccel<BlockStructuredCUBQLSampler>::isProg();
  }
  
  OPTIX_CLOSEST_HIT_PROGRAM(BlockStructured_MCDDA_CH)()
  {
    MCDDAVolumeAccel<BlockStructuredCUBQLSampler>::chProg();
  }

}
#endif
