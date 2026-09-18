// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/geometry/Geometry.h"

namespace BARNEY_NS {
  namespace native {

    // ==================================================================
    /*! Scalar field made of 3D structured data, constting of Nx*Ny*Nz
      scalars.

      Supported settable fields:

      - "vertices"  (BNData<float3>)
      - "indices"   (BNData<int3>)
      - "texcoords" (BNData<float2>)

      Per-vertex normals arrive through the ANARI attribute system
      (vertex.normal -> normalAttribute), not a bespoke array.
    */
    struct Triangles : public Geometry {
      typedef std::shared_ptr<Triangles> SP;

      struct DD : public Geometry::DD {
        const vec3i *indices;
        const vec3f *vertices;
        const vec2f *texcoords;
        // const vec4f *vertexAttribute[5];
      };
    
      Triangles(Context *context, DevGroup::SP devices);
      virtual ~Triangles();
    
      /*! pretty-printer for printf-debugging */
      std::string toString() const override
      { return "Triangles{}"; }
    
      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      void commit() override;
      bool setData(const std::string &member,
                   const Data::SP &value) override;
      /*! @} */
      // ------------------------------------------------------------------

      PODData::SP vertices;
      PODData::SP indices;
      // TODO: do we still need this in times of ANARI?
      PODData::SP texcoords;
    };

  }
}

