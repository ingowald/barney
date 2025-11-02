// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/common/math.h"
#include "barney/light/Light.h"

namespace BARNEY_NS {

  struct PointLight : public Light {
    struct DD : public Light::DD {
      inline __rtc_device vec3f radianceTowards(vec3f P) const
      {
        if (isnan(intensity))
          return color * power * ONE_OVER_FOUR_PI;
        else
          return color * intensity;
      }
      vec3f position;
      float intensity;
      float power;
    };
    
    typedef std::shared_ptr<PointLight> SP;
    PointLight(Context *context,
               const DevGroup::SP &devices)
      : Light(context,devices)
    {}
    
    DD getDD(const affine3f &instanceXfm) const;
    
    std::string toString() const override { return "PointLight"; }
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    vec3f position = vec3f(0.f,0.f,0.f);
    float power = 1.f;
    float intensity = NAN;
  };

}
