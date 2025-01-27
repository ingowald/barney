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
#include "barney/volume/MCAccelerator.h"

RTC_DECLARE_GLOBALS(barney::render::OptixGlobals);

namespace barney {

  // struct Structured_MCRTX_Programs {
  //   template<typename RTBackend>
  //   static inline __both__
  //   void bounds(const RTBackend &rt,
  //               const void *geomData,
  //               owl::common::box3f &bounds,  
  //               const int32_t primID)
  //   {printf("todo\n");}
  //   template<typename RTBackend>
  //   static inline __both__
  //   void closest_hit(const RTBackend &rt)
  //   {printf("todo\n");}
  //   template<typename RTBackend>
  //   static inline __both__
  //   void any_hit(const RTBackend &rt)
  //   {printf("todo\n");}
  //   template<typename RTBackend>
  //   static inline __both__
  //   void intersect(const RTBackend &rt)
  //   {printf("todo\n");}
  // };
  struct MCAccel_Structured_Programs {
    
    template<typename TraceInterface>
    static inline __both__
    void bounds(const TraceInterface &ti,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    {
      MCVolumeAccel<StructuredData>::boundsProg(ti,geomData,bounds,primID);
      // const MCVolumeAccel<StructuredData>::DD &geom
      //   = *(const MCVolumeAccel<StructuredData>::DD *)geomData;
      // bounds = geom.volume.sf.worldBounds;
    }
    
    template<typename TraceInterface>
    static inline __both__
    void intersect(TraceInterface &ti)
    {
      MCVolumeAccel<StructuredData>::isProg(ti);
      // const void *geomData = ti.getGeomData();
      // const MCVolumeAccel<StructuredData>::DD &geom
      //   = *(const MCVolumeAccel<StructuredData>::DD *)geomData;
      // geom.isProg(ti);
    }
    
    template<typename TraceInterface>
    static inline __both__
    void closest_hit(TraceInterface &ti)
    { /* nothing to do */ }
    
    template<typename TraceInterface>
    static inline __both__
    void any_hit(TraceInterface &ti)
    { /* nothing to do */ }
  };
}

RTC_DECLARE_USER_GEOM(MCAccel_Structured,barney::MCAccel_Structured_Programs);

