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

#include "rtcore/embree/GeomType.h"
#include "rtcore/embree/Buffer.h"

namespace rtc {
  namespace embree {

    struct Geom
    {
      Geom(GeomType *type);
      virtual ~Geom() = default;
      void setDD(const void *dd);

      /*! only for user geoms */
      virtual void setPrimCount(int primCount) = 0;
      /*! can only get called on triangle type geoms */
      virtual void setVertices(Buffer *vertices, int numVertices) = 0;
      virtual void setIndices(Buffer *indices, int numIndices) = 0;
      
      std::vector<int8_t> programData;
      GeomType *const type;
    };
    
  }
}
