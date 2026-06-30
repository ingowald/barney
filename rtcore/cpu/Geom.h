// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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
