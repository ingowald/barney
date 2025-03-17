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

#include "rtcore/cuda/Device.h"
#include "rtcore/cuda/Buffer.h"

namespace rtc {
  namespace cuda {

    struct GeomType;
    
    struct Geom {
      Geom(GeomType *gt);
      virtual ~Geom();
      void setDD(const void *dd);

      virtual void setPrimCount(int primCount) = 0;
      virtual void setVertices(Buffer *vertices, int numVertices) = 0;
      virtual void setIndices(Buffer *indices, int numIndices) = 0;
      
      GeomType *const gt;
    };

    struct TrianglesGeom : public cuda::Geom {
      TrianglesGeom(GeomType *gt);
      
      void setPrimCount(int primCount) override { assert(0); }
      void setVertices(Buffer *vertices, int numVertices) override;
      void setIndices(Buffer *indices, int numIndices) override;

      Buffer *vertices = 0;
      int numVertices = 0;
      Buffer *indices = 0;
      int numIndices = 0;
    };
    struct UserGeom : public cuda::Geom {
      UserGeom(GeomType *gt);
      
      void setPrimCount(int primCount);
      void setVertices(Buffer *vertices, int numVertices) override
      { assert(0); }
      void setIndices(Buffer *indices, int numIndices) override
      { assert(0); }
    };
    
  }
}
