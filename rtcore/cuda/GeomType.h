// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/cudaCommon/Device.h"

namespace rtc {
  namespace cuda {
  
    struct Geom;
    struct TraceInterface;
    
    typedef void (*BoundsKernel)(Device *device,
                                 const void *dd,
                                 box3f *boundsArray,
                                 int numPrims);
    typedef void (*IntersectProg)(TraceInterface &ti);
    typedef void (*AHProg)(TraceInterface &ti);
    typedef void (*CHProg)(TraceInterface &ti);
    
    struct GeomType
    {
      GeomType(Device *device, size_t sizeOfDD);
      virtual ~GeomType() = default;
      
      virtual Geom *createGeom() = 0;
               
      Device *const device;
      size_t  const sizeOfDD;
    };

    struct UserGeomType : public GeomType {
      UserGeomType(Device *device,
                   size_t sizeOfDD,
                   // BoundsProg bounds,
                   BoundsKernel bounds,
                   IntersectProg intersect,
                   AHProg ah,
                   CHProg ch);
      virtual ~UserGeomType() = default;
      Geom *createGeom() override;

      BoundsKernel const bounds;
      // BoundsProg const bounds;
      IntersectProg const intersect;
      AHProg const ah;
      CHProg const ch;
    };

    struct TrianglesGeomType : public GeomType {
      TrianglesGeomType(Device *device,
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





