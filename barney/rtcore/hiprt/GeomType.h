// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>

#pragma once

#include "rtcore/cudaCommon/Device.h"

namespace rtc {
  namespace hiprt {

    struct Geom;
    struct Device;
    struct TraceInterface;

    typedef void (*BoundsKernel)(Device *device,
                                 const void *dd,
                                 box3f *boundsArray,
                                 int numPrims);
    typedef void (*IntersectProg)(TraceInterface &ti);
    typedef void (*AHProg)(TraceInterface &ti);
    typedef void (*CHProg)(TraceInterface &ti);

    struct GeomType {
      GeomType(cuda_common::Device *device, size_t sizeOfDD);
      virtual ~GeomType() = default;
      virtual Geom *createGeom() = 0;

      cuda_common::Device *const device;
      size_t  const sizeOfDD;
    };

    struct UserGeomType : public GeomType {
      UserGeomType(cuda_common::Device *device,
                   size_t sizeOfDD,
                   BoundsKernel bounds,
                   IntersectProg intersect,
                   AHProg ah,
                   CHProg ch);
      virtual ~UserGeomType() = default;
      Geom *createGeom() override;

      BoundsKernel  const bounds;
      IntersectProg const intersect;
      AHProg const ah;
      CHProg const ch;
    };

    struct TrianglesGeomType : public GeomType {
      TrianglesGeomType(cuda_common::Device *device,
                        size_t sizeOfDD,
                        AHProg ah,
                        CHProg ch);
      virtual ~TrianglesGeomType() = default;
      Geom *createGeom() override;

      AHProg const ah;
      CHProg const ch;
    };

  }
}
