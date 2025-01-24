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

#include "barney/geometry/Attributes.dev.h"
#include "barney/volume/StructuredData.h"

RTC_DECLARE_GLOBALS(barney::render::OptixGlobals);

namespace barney {

  struct Structured_MCRTX_Programs {
    template<typename RTBackend>
    static inline __both__
    void bounds(const RTBackend &rt,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    {printf("todo\n");}
    template<typename RTBackend>
    static inline __both__
    void closest_hit(const RTBackend &rt)
    {printf("todo\n");}
    template<typename RTBackend>
    static inline __both__
    void any_hit(const RTBackend &rt)
    {printf("todo\n");}
    template<typename RTBackend>
    static inline __both__
    void intersect(const RTBackend &rt)
    {printf("todo\n");}
  };
  struct Structured_MCDDA_Programs {
    template<typename RTBackend>
    static inline __both__
    void bounds(const RTBackend &rt,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    {printf("todo\n");}
    template<typename RTBackend>
    static inline __both__
    void closest_hit(const RTBackend &rt)
    {printf("todo\n");}
    template<typename RTBackend>
    static inline __both__
    void any_hit(const RTBackend &rt)
    {printf("todo\n");}
    template<typename RTBackend>
    static inline __both__
    void intersect(const RTBackend &rt)
    {printf("todo\n");}
  };
#if 0
  OPTIX_BOUNDS_PROGRAM(Structured_MCRTX_Bounds)(const void *geomData,
                                                owl::common::box3f &bounds,
                                                const int32_t primID)
  {
    MCRTXVolumeAccel<StructuredData>::boundsProg
      (geomData,bounds,primID);
  }

  OPTIX_INTERSECT_PROGRAM(Structured_MCRTX_Isec)()
  {
    MCRTXVolumeAccel<StructuredData>::isProg();
  }
  
  OPTIX_CLOSEST_HIT_PROGRAM(Structured_MCRTX_CH)()
  {
    /* nothing - already all set in isec */
    MCRTXVolumeAccel<StructuredData>::chProg();
  }
  





  OPTIX_BOUNDS_PROGRAM(Structured_MCDDA_Bounds)(const void *geomData,
                                                owl::common::box3f &bounds,
                                                const int32_t primID)
  {
    MCDDAVolumeAccel<StructuredData>::boundsProg(geomData,bounds,primID);
  }

  OPTIX_INTERSECT_PROGRAM(Structured_MCDDA_Isec)()
  {
    MCDDAVolumeAccel<StructuredData>::isProg();
  }
  
  OPTIX_CLOSEST_HIT_PROGRAM(Structured_MCDDA_CH)()
  {
    MCDDAVolumeAccel<StructuredData>::chProg();
    /* nothing - already all set in isec */
  }
#endif
}

RTC_DECLARE_USER_GEOM(Structured_MCRTX,barney::Structured_MCRTX_Programs);
RTC_DECLARE_USER_GEOM(Structured_MCDDA,barney::Structured_MCDDA_Programs);

