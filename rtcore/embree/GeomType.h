// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/embree/Device.h"

namespace BARNEY_NS {
  namespace rtc {

    struct TraceInterface;
    struct Device;
    
    typedef void (*AnyHitFct)(TraceInterface &ti);
    typedef void (*ClosestHitFct)(TraceInterface &ti);
    typedef void (*IntersectFct)(TraceInterface &ti);
    typedef void (*BoundsFct)(const TraceInterface &ti,
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
  extern BARNEY_NS::rtc::GeomType *                                     \
  createGeomType_##typeName(BARNEY_NS::rtc::Device *);


#define RTC_IMPORT_TRIANGLES_GEOM(moduleName,typeName,DD,has_ah,has_ch) \
  extern BARNEY_NS::rtc::GeomType *                                     \
  createGeomType_##typeName(BARNEY_NS::rtc::Device *);


#define RTC_EXPORT_USER_GEOM(name,DD,Programs,has_ah,has_ch)    \
  BARNEY_NS::rtc::GeomType *                                    \
  createGeomType_##name(BARNEY_NS::rtc::Device *device)         \
  {                                                             \
    return new BARNEY_NS::rtc::UserGeomType                     \
      (device,                                                  \
       sizeof(DD),                                              \
       Programs::bounds,                                        \
       Programs::intersect,                                     \
       has_ah?Programs::anyHit:0,                               \
       has_ch?Programs::closestHit:0);                          \
  }


#define RTC_EXPORT_TRIANGLES_GEOM(name,DD,Programs,has_ah,has_ch)       \
  BARNEY_NS::rtc::GeomType *                                            \
  createGeomType_##name(BARNEY_NS::rtc::Device *device)                 \
  {                                                                     \
    return new BARNEY_NS::rtc::TrianglesGeomType                        \
      (device,                                                          \
       sizeof(DD),                                                      \
       has_ah?Programs::anyHit:0,                                       \
       has_ch?Programs::closestHit:0);                                  \
  }




