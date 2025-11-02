// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/geometry/Geometry.h"

namespace BARNEY_NS {

  struct Spheres : public Geometry {
    typedef std::shared_ptr<Spheres> SP;

    struct DD : public Geometry::DD {
      vec3f       *origins;
      float       *radii;
      vec3f       *colors;
      float        defaultRadius;
      // const vec4f *vertexAttribute[5];
    };

    Spheres(Context *context, DevGroup::SP devices);
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Spheres{}"; }

    void commit() override;
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool set1f(const std::string &member, const float &value) override;
    bool setData(const std::string &member, const barney_api::Data::SP &value) override;
    bool setObject(const std::string &member, const Object::SP &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    PODData::SP origins = 0;
    PODData::SP colors  = 0;
    PODData::SP radii   = 0;
    float       defaultRadius = .1f;
  };
  
}
  
