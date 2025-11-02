// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "rtcore/embree/Triangles.h"

namespace rtc {
  namespace embree {

    TrianglesGeom::TrianglesGeom(TrianglesGeomType *type)
      : Geom(type)
    {}
    
    /*! only for user geoms */
    void TrianglesGeom::setPrimCount(int primCount)
    {
      throw std::runtime_error("setPrimCount only makes sense for user geoms");
    }
    
    /*! can only get called on triangle type geoms */
    void TrianglesGeom::setVertices(Buffer *vertices,
                                    int numVertices)
    {
      this->vertices = (vec3f*)((Buffer *)vertices)->mem;
      this->numVertices = numVertices;
    }
    
    void TrianglesGeom::setIndices(Buffer *indices, int numIndices)
    {
      this->indices = (vec3i*)((Buffer *)indices)->mem;
      this->numIndices = numIndices;
    }

  }
}
