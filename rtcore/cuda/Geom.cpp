// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "rtcore/cuda/Device.h"
#include "rtcore/cuda/Geom.h"
#include "rtcore/cuda/GeomType.h"

namespace rtc {
  namespace cuda {
    
    Geom::Geom(GeomType *gt)
      : gt(gt),
        data(gt->sizeOfDD)
    {}
    
    Geom::~Geom()
    {}
    
    void Geom::setDD(const void *dd)
    {
      memcpy(data.data(),dd,data.size());
    }

    TrianglesGeom::TrianglesGeom(GeomType *gt)
      : Geom(gt)
    {}
      
    void TrianglesGeom::setVertices(Buffer *vertices,
                                    int numVertices)
    {
      this->vertices = vertices;
      this->numVertices = numVertices;
    }
      
    void TrianglesGeom::setIndices(Buffer *indices,
                                   int numIndices)
    {
      this->indices = indices;
      this->numIndices = numIndices;
    }


    UserGeom::UserGeom(GeomType *gt)
      : Geom(gt)
    {}
      
    void UserGeom::setPrimCount(int primCount)
    {
      this->primCount = primCount;
    }

  }
}
