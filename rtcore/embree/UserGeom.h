// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/embree/Buffer.h"
#include "rtcore/embree/Geom.h"

namespace rtc {
  namespace embree {

    struct UserGeom : public Geom
    {
      UserGeom(UserGeomType *type);
      
      /*! only for user geoms */
      void setPrimCount(int primCount) override;
      /*! can only get called on triangle type geoms */
      void setVertices(Buffer *vertices, int numVertices) override;
      void setIndices(Buffer *indices, int numIndices) override;

      int primCount = 0;
    };
    
  }
}


