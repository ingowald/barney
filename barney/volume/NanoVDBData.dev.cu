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

#include "barney/volume/NanoVDBData.h"

namespace barney {

  OPTIX_BOUNDS_PROGRAM(NanoVDB_MCRTX_Bounds)(const void *geomData,
                                                owl::common::box3f &bounds,
                                                const int32_t primID)
  {
    MCRTXVolumeAccel<NanoVDBDataSampler>::boundsProg
      (geomData,bounds,primID);
  }

  OPTIX_INTERSECT_PROGRAM(NanoVDB_MCRTX_Isec)()
  {
    MCRTXVolumeAccel<NanoVDBDataSampler>::isProg();
  }
  
  OPTIX_CLOSEST_HIT_PROGRAM(NanoVDB_MCRTX_CH)()
  {
    /* nothing - already all set in isec */
    MCRTXVolumeAccel<NanoVDBDataSampler>::chProg();
  }

  OPTIX_BOUNDS_PROGRAM(NanoVDB_MCDDA_Bounds)(const void *geomData,
                                                owl::common::box3f &bounds,
                                                const int32_t primID)
  {
    MCDDAVolumeAccel<NanoVDBDataSampler>::boundsProg(geomData,bounds,primID);
  }

  OPTIX_INTERSECT_PROGRAM(NanoVDB_MCDDA_Isec)()
  {
    MCDDAVolumeAccel<NanoVDBDataSampler>::isProg();
  }
  
  OPTIX_CLOSEST_HIT_PROGRAM(NanoVDB_MCDDA_CH)()
  {
    MCDDAVolumeAccel<NanoVDBDataSampler>::chProg();
    /* nothing - already all set in isec */
  }
  
}

