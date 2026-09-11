// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/cpu/Buffer.h"
#include "rtcore/cpu/GeomType.h"
#include "rtcore/cpu/Geom.h"

namespace BARNEY_NS {
  namespace rtc {

    struct TrianglesGeomType;
    
    struct TrianglesGeom : public Geom
    {
      TrianglesGeom(TrianglesGeomType *type);

      /*! only for user geoms */
      void setPrimCount(int primCount) override;
      /*! can only get called on triangle type geoms */
      void setVertices(Buffer *vertices, int numVertices) override;
      void setIndices(Buffer *indices, int numIndices) override;

      int numVertices = 0;
      int numIndices = 0;
      vec3f *vertices = 0;
      vec3i *indices = 0;
    };
    
  }
}


