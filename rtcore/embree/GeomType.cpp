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
