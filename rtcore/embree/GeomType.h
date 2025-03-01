// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

#pragma once

#include "rtcore/embree/Device.h"

namespace rtc {
  namespace embree {
    struct TraceInterface;
    struct Device;
    
    typedef void (*AnyHitFct)(embree::TraceInterface &ti);
    typedef void (*ClosestHitFct)(embree::TraceInterface &ti);
    typedef void (*IntersectFct)(embree::TraceInterface &ti);
    typedef void (*BoundsFct)(const embree::TraceInterface &ti,
                              const void *gt, box3f &result, int primID);


    struct GeomType
    {
      GeomType(Device       *device,
               size_t        sizeOfProgramData,
               AnyHitFct     ah,
               ClosestHitFct ch);
      virtual ~GeomType() = default;
      virtual Geom *createGeom() = 0;
      
      AnyHitFct     const ah;
      ClosestHitFct const ch;
      size_t        const sizeOfProgramData;
      Device       *const device;
    };
      
    struct TrianglesGeomType : public GeomType
    {
      TrianglesGeomType(Device       *device,
                        size_t        sizeOfProgramData,
                        AnyHitFct     ah,
                        ClosestHitFct ch);
      
      virtual ~TrianglesGeomType();
      
      Geom *createGeom() override;
    };

    struct UserGeomType : public GeomType
    {
      UserGeomType(Device       *device,
                   size_t        sizeOfProgramData,
                   BoundsFct     bounds,
                   IntersectFct  intersect,
                   AnyHitFct     ah,
                   ClosestHitFct ch);
      
      virtual ~UserGeomType();
      
      Geom *createGeom() override;
      
      BoundsFct     const bounds;
      IntersectFct  const intersect;
    };
    
  }
}

#define RTC_IMPORT_USER_GEOM(moduleName,typeName,DD,has_ah,has_ch)      \
  extern ::rtc::GeomType *createGeomType_##typeName(::rtc::Device *);

#define RTC_IMPORT_TRIANGLES_GEOM(moduleName,typeName,DD,has_ah,has_ch) \
  extern rtc::GeomType *createGeomType_##typeName(rtc::Device *);



#define RTC_EXPORT_USER_GEOM(name,DD,Programs,has_ah,has_ch)    \
  ::rtc::GeomType *createGeomType_##name(::rtc::Device *device) \
  {                                                             \
    return new ::rtc::embree::UserGeomType                      \
      (device,                                                  \
       sizeof(DD),                                              \
       Programs::bounds,                                        \
       Programs::intersect,                                     \
       has_ah?Programs::anyHit:0,                               \
       has_ch?Programs::closestHit:0);                          \
  }


#define RTC_EXPORT_TRIANGLES_GEOM(name,DD,Programs,has_ah,has_ch)       \
  rtc::GeomType *createGeomType_##name(rtc::Device *device)             \
  {                                                                     \
    return new rtc::embree::TrianglesGeomType                           \
      (device,                                                          \
       sizeof(DD),                                                      \
       has_ah?Programs::anyHit:0,                                       \
       has_ch?Programs::closestHit:0);                                  \
  }




