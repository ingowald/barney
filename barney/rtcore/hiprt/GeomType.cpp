// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>

#include "rtcore/hiprt/GeomType.h"
#include "rtcore/hiprt/Geom.h"

namespace rtc {
  namespace hiprt {

    GeomType::GeomType(cuda_common::Device *device, size_t sizeOfDD)
      : device(device), sizeOfDD(sizeOfDD)
    {}

    UserGeomType::UserGeomType(cuda_common::Device *device,
                               size_t sizeOfDD,
                               BoundsKernel bounds,
                               IntersectProg intersect,
                               AHProg ah,
                               CHProg ch)
      : GeomType(device,sizeOfDD),
        bounds(bounds), intersect(intersect), ah(ah), ch(ch)
    {}

    TrianglesGeomType::TrianglesGeomType(cuda_common::Device *device,
                                         size_t sizeOfDD,
                                         AHProg ah,
                                         CHProg ch)
      : GeomType(device,sizeOfDD), ah(ah), ch(ch)
    {}

    Geom *UserGeomType::createGeom()      { return new UserGeom(this); }
    Geom *TrianglesGeomType::createGeom() { return new TrianglesGeom(this); }

  }
}
