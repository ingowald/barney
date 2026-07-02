// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rtcore/embree/UserGeom.h"

namespace BARNEY_NS {
  namespace rtc {

    UserGeom::UserGeom(UserGeomType *type)
      : Geom(type)
    {}
    
    /*! only for user geoms */
    void UserGeom::setPrimCount(int primCount)
    {
      this->primCount = primCount;
    }
    
    /*! can only get called on triangle type geoms */
    void UserGeom::setVertices(Buffer *vertices,
                                    int numVertices)
    {/*ignore*/}
    void UserGeom::setIndices(Buffer *indices, int numIndices)
    {/*ignore*/}
    
  }
}
