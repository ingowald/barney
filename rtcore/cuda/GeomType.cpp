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

#include "rtcore/cuda/Device.h"
#include "rtcore/cuda/GeomType.h"
#include "rtcore/cuda/Geom.h"

namespace rtc {
  namespace cuda {

    GeomType::GeomType(Device *device,
                       size_t sizeOfDD)
      : device(device), sizeOfDD(sizeOfDD)
    {
    }
    
    UserGeomType::UserGeomType(Device *device,
                   size_t sizeOfDD,
                   BoundsKernel bounds,
                   // BoundsProg bounds,
                   IntersectProg intersect,
                   AHProg ah,
                   CHProg ch)
      : GeomType(device,sizeOfDD),
        bounds(bounds),
        intersect(intersect),
        ah(ah),
        ch(ch)
    {}
    
    TrianglesGeomType::TrianglesGeomType(Device *device,
                        size_t sizeOfDD,
                        AHProg ah,
                        CHProg ch)
      : GeomType(device,sizeOfDD),
        ah(ah),
        ch(ch)
    {}

    Geom *UserGeomType::createGeom()
    {
      return new UserGeom(this);
    }
    
    Geom *TrianglesGeomType::createGeom()
    {
      return new TrianglesGeom(this);
    }

  }
}
