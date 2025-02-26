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

#include "rtcore/embree/Buffer.h"
#include "rtcore/embree/GeomType.h"
#include "rtcore/embree/Geom.h"

namespace rtc {
  namespace embree {

    struct TrianglesGeomType;
    
    struct TrianglesGeom : public Geom
    {
      TrianglesGeom(TrianglesGeomType *type);

      /*! only for user geoms */
      void setPrimCount(int primCount) override;
      /*! can only get called on triangle type geoms */
      void setVertices(Buffer *vertices, int numVertices) override;
      void setIndices(Buffer *indices, int numIndices) override;

      vec3f *vertices = 0;
      int numVertices = 0;
      vec3i *indices = 0;
      int numIndices = 0;
    };
    
  }
}


