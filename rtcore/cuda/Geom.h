// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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
