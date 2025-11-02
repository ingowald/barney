// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/geometry/Geometry.h"

namespace BARNEY_NS {

  /*! A geometry made of multiple "capsules", where each capsule is
      "pill-like" shape obtained by linearly connecting two
      spheres. Unlike cylinders both end-points of the capsule have
      their own radius that the rest of the shape linearly
      interpolates between; capsules also always have "rounded caps"
      in the sense that both of the end points form complete
      spheres. Capsules can also interpolate a "color" attribute.

      Is defined by three parameters:

      `int2 radii[]` two vertex indices per prim, specifying end point
      position and radii for each capsule.

      `float3 vertices[]` position (.xyz) and radius (.w) of each vertex
  */
  struct Capsules : public Geometry {
    typedef std::shared_ptr<Capsules> SP;

    struct DD : public Geometry::DD {
      const vec4f *vertices;
      const vec2i *indices;
    };
    
    Capsules(Context *context, DevGroup::SP devices);
    virtual ~Capsules() = default;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Capsules{}"; }
    
    void commit() override;
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool setData(const std::string &member, const barney_api::Data::SP &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    PODData::SP vertices;
    PODData::SP indices;
  };

}
