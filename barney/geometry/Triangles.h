// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/geometry/Geometry.h"

namespace BARNEY_NS {

  struct ModelSlot;


  // ==================================================================
  /*! Scalar field made of 3D structured data, constting of Nx*Ny*Nz
      scalars.

      Supported settable fields:

      - "vertices"  (BNData<float3>)
      - "indices"   (BNData<int3>)
      - "normals"   (BNData<float3>)
      - "texcoords" (BNData<float2>)
  */
  struct Triangles : public Geometry {
    typedef std::shared_ptr<Triangles> SP;

    struct DD : public Geometry::DD {
      const vec3i *indices;
      const vec3f *vertices;
      const vec3f *normals;
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
                 const barney_api::Data::SP &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    PODData::SP vertices;
    PODData::SP indices;
    PODData::SP normals;
    // TODO: do we still need this in times of ANARI?
    PODData::SP texcoords;
  };

}
