// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/light/Light.h"

namespace BARNEY_NS {

  struct DirLight : public Light {
    struct DD {
      vec3f direction;
      float radiance;
      vec3f color;
    };
    
    typedef std::shared_ptr<DirLight> SP;
    DirLight(Context *context,
             const DevGroup::SP &devices)
      : Light(context,devices)
    {}
    
    DD getDD(const affine3f &instanceXfm) const;
    
    std::string toString() const override { return "DirLight"; }
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    /*! SPEC: main emission direction of the directional light */
    vec3f direction{0.f, 0.f, -1.f};
    
    /*! SPEC: the amount of light arriving at a surface point,
        assuming the light is oriented towards to the surface, in
        W/m2 */
    float irradiance = NAN;
    /*! the amount of light emitted in a direction, in W/sr/m2;
        irradiance takes precedence if also specified */
    float radiance = 1.f;
  };

}
