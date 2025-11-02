// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "rtcore/cuda/Device.h"
#include "rtcore/cuda/GeomType.h"
#include "rtcore/cuda/Geom.h"

namespace rtc {
  namespace cuda {
    
    GeomType::GeomType(Device *device,
                       size_t sizeOfDD)
      : device(device), sizeOfDD(sizeOfDD)
    {}
    
    UserGeomType::UserGeomType(Device *device,
                               size_t sizeOfDD,
                               BoundsKernel bounds,
                               // BoundsProg bounds,
                               IntersectProg intersect,
                               AHProg ah,
                               CHProg ch)
      : GeomType(device,sizeOfDD),
        bounds(bounds),
        intersect(intersect),
        ah(ah),
        ch(ch)
    {}
    
    TrianglesGeomType::TrianglesGeomType(Device *device,
                                         size_t sizeOfDD,
                                         AHProg ah,
                                         CHProg ch)
      : GeomType(device,sizeOfDD),
        ah(ah),
        ch(ch)
    {}

    Geom *UserGeomType::createGeom()
    {
      return new UserGeom(this);
    }
    
    Geom *TrianglesGeomType::createGeom()
    {
      return new TrianglesGeom(this);
    }

  }
}
