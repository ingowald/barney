// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/optix/Device.h"
#include "rtcore/optix/Buffer.h"
#include <owl/owl.h>

namespace rtc {
  namespace optix {
    struct Device;
    
    struct GeomType;
    
    struct Geom {
      Geom(GeomType *gt, OWLGeom geom);
      virtual ~Geom();
      void setDD(const void *dd);

      virtual void setPrimCount(int primCount) = 0;
      virtual void setVertices(Buffer *vertices, int numVertices) = 0;
      virtual void setIndices(Buffer *indices, int numIndices) = 0;
      
      GeomType *const gt;
      OWLGeom   const owl;
    };

    struct TrianglesGeom : public optix::Geom {
      TrianglesGeom(GeomType *gt, OWLGeom geom);
      
      void setPrimCount(int primCount) override { assert(0); }
      void setVertices(Buffer *vertices, int numVertices) override;
      void setIndices(Buffer *indices, int numIndices) override;
      /*! only for user geoms */
      /*! can only get called on triangle type geoms */
    };
    struct UserGeom : public optix::Geom {
      UserGeom(GeomType *gt, OWLGeom geom);
      
      void setPrimCount(int primCount);
      void setVertices(Buffer *vertices, int numVertices) override
      { assert(0); }
      void setIndices(Buffer *indices, int numIndices) override
      { assert(0); }
    };
    
    struct GeomType {
      GeomType(optix::Device *device);
      virtual ~GeomType();
      
      virtual Geom *createGeom() = 0;
      
      OWLGeomType gt = 0;
      optix::Device *const device;
    };
    struct TrianglesGeomType : public GeomType
    {
      TrianglesGeomType(optix::Device *device,
                        const std::string &ptxCode,
                        const std::string &typeName,
                        size_t sizeOfDD,
                        bool has_ah, bool has_ch);
      Geom *createGeom() override;
    };
    struct UserGeomType : public GeomType
    {
      UserGeomType(optix::Device *device,
                   const std::string &ptxCode,
                   const std::string &typeName,
                   size_t sizeOfDD,
                   bool has_ah, bool has_ch);
      Geom *createGeom() override;
    };
    
  }
}
