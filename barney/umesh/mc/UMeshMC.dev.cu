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

/*! \file UMeshCUBQLMC.dev.cu implements the DDA and RTX traversers for a umesh
  scalar field with cubql sampler and macro cell accel */

#include "barney/umesh/mc/UMeshCUBQLSampler.h"
#include "barney/volume/DDA.h"
// #include <owl/owl_device.h>

RTC_DECLARE_GLOBALS(barney::render::OptixGlobals);

namespace barney {
#if 0
  // ==================================================================
  //
  // UMesh Data, Macro-Cell (MC) accelerator, and traversal
  // via an RTX BVH built over active macro-cells
  //
  // ================================================================== 

  OPTIX_BOUNDS_PROGRAM(UMesh_CUBQL_MCRTX_Bounds)(const void *geomData,
                                                 owl::common::box3f &bounds,
                                                 const int32_t primID)
  {
    MCRTXVolumeAccel<UMeshCUBQLSampler>::boundsProg
      (geomData,bounds,primID);
  }

  OPTIX_INTERSECT_PROGRAM(UMesh_CUBQL_MCRTX_Isec)()
  {
    MCRTXVolumeAccel<UMeshCUBQLSampler>::isProg();
  }
  
  OPTIX_CLOSEST_HIT_PROGRAM(UMesh_CUBQL_MCRTX_CH)()
  {
    MCRTXVolumeAccel<UMeshCUBQLSampler>::chProg();
  }
  




  // ==================================================================
  //
  // UMesh Data, Macro-Cell (MC) accelerator, and cuda-DDA
  // traversal
  //
  // ================================================================== 

  OPTIX_BOUNDS_PROGRAM(UMesh_CUBQL_MCDDA_Bounds)(const void *geomData,
                                                 owl::common::box3f &bounds,
                                                 const int32_t primID)
  {
    MCDDAVolumeAccel<UMeshCUBQLSampler>::boundsProg(geomData,bounds,primID);
  }

  OPTIX_INTERSECT_PROGRAM(UMesh_CUBQL_MCDDA_Isec)()
  {
    MCDDAVolumeAccel<UMeshCUBQLSampler>::isProg();
  }
  
  OPTIX_CLOSEST_HIT_PROGRAM(UMesh_CUBQL_MCDDA_CH)()
  {
    MCDDAVolumeAccel<UMeshCUBQLSampler>::chProg();
    /* nothing - already all set in isec */
  }
#endif  
}

