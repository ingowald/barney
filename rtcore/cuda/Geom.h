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
#include "rtcore/cuda/GeomType.h"

namespace rtc {
  namespace cuda {
    
    struct GeomType;
    
    struct Geom {
      Geom(GeomType *gt);
      virtual ~Geom();
      void setDD(const void *dd);

      struct SBTHeader {
        AHProg ah;
        CHProg ch;
        union {
          struct {
            const vec3f *vertices;
            const vec3i *indices;
          } triangles;
          struct {
            // boundsprog is axed - we handle bounds as a complete kernel
            // jujst like for optix backend.
            //BoundsProg    bounds;
            IntersectProg intersect;
          } user;
        };
      };
      
      virtual void setPrimCount(int primCount) = 0;
      virtual void setVertices(Buffer *vertices, int numVertices) = 0;
      virtual void setIndices(Buffer *indices, int numIndices) = 0;
      
      GeomType *const gt;
      std::vector<uint8_t> data;
    };

    struct TrianglesGeom : public Geom {
      TrianglesGeom(GeomType *gt);
      
      void setPrimCount(int primCount) override { assert(0); }
      void setVertices(Buffer *vertices, int numVertices) override;
      void setIndices(Buffer *indices, int numIndices) override;

      Buffer *vertices = 0;
      int numVertices = 0;
      Buffer *indices = 0;
      int numIndices = 0;
    };
  
    struct UserGeom : public Geom {
      UserGeom(GeomType *gt);
      
      void setPrimCount(int primCount) override;
      void setVertices(Buffer *vertices, int numVertices) override
      { assert(0); }
      void setIndices(Buffer *indices, int numIndices) override
      { assert(0); }

      int primCount = 0;
    };
    
  }
}
