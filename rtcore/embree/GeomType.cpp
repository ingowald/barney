// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "rtcore/embree/GeomType.h"
#include "rtcore/embree/Triangles.h"
#include "rtcore/embree/UserGeom.h"

namespace rtc {
  namespace embree {

    // GeomType::GeomType(Device *device,
    //                    const std::string &name,
    //                    size_t sizeOfProgramData,
    //                    bool has_ah,
    //                    bool has_ch)
    //   : sizeOfProgramData(sizeOfProgramData)
    // {
    //   if (has_ah)
    //     ah = (AnyHitFct)rtc::getSymbol("barney_embree_ah_"+name);
    //   if (has_ch)
    //     ch = (ClosestHitFct)rtc::getSymbol("barney_embree_ch_"+name);
    // }
    
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

    // UserGeomType::UserGeomType(Device *device,
    //                                      const std::string &name,
    //                                      size_t sizeOfProgramData,
    //                                      bool has_ah,
    //                                      bool has_ch)
    //   : GeomType(device,name,sizeOfProgramData,has_ah,has_ch)
    // {
    //   intersect = (IntersectFct)rtc::getSymbol("barney_embree_intersect_"+name);
    //   bounds    = (BoundsFct)rtc::getSymbol("barney_embree_bounds_"+name);
    // }

    
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
