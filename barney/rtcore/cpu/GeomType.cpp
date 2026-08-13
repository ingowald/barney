// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rtcore/cpu/GeomType.h"
#include "rtcore/cpu/Triangles.h"
#include "rtcore/cpu/UserGeom.h"

namespace BARNEY_NS {
  namespace rtc {

    GeomType::GeomType(Device *device,
                       size_t sizeOfProgramData,
                       AnyHitFct     ah,
                       ClosestHitFct ch)
      : device(device),
        sizeOfProgramData(sizeOfProgramData),
        ah(ah),ch(ch)
    {}
    
    TrianglesGeomType::TrianglesGeomType(Device *device,
                                         size_t sizeOfProgramData,
                                         AnyHitFct     ah,
                                         ClosestHitFct ch)
      : GeomType(device,sizeOfProgramData,ah,ch)
    {
    }
    
    UserGeomType::UserGeomType(Device *device,
                               size_t sizeOfProgramData,
                               BoundsFct     bounds,
                               IntersectFct  intersect,
                               AnyHitFct     ah,
                               ClosestHitFct ch)
      : GeomType(device,sizeOfProgramData,ah,ch),
        bounds(bounds),
        intersect(intersect)
    {
    }

    UserGeomType::~UserGeomType()
    {}
    
    TrianglesGeomType::~TrianglesGeomType()
    {}
    
    Geom *TrianglesGeomType::createGeom()
    { return new TrianglesGeom(this); }

    Geom *UserGeomType::createGeom()
    { return new UserGeom(this); }
    
  }
}
