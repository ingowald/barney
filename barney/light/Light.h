// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/Object.h"
#include "barney/common/Data.h"
#include "barney/common/Texture.h"

namespace BARNEY_NS {

  struct ModelSlot;
  
  struct Light : public barney_api::Light {
    typedef std::shared_ptr<Light> SP;

    struct DD {
      vec3f color;
    };
    
    /*! what we return, during rendering, when we sample a light
        source */
    struct Sample {
      /* direction _to_ light */
      vec3f direction;
      /*! radiance coming _from_ dir */
      vec3f radiance;
      /*! distance to this light sample */
      float distance;
      /*! pdf of sample that was chosen */
      float pdf = 0.f;
    };
  
    
    Light(Context *context, const DevGroup::SP &devices);

    std::string toString() const override { return "Light<>"; }

    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    // bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    
    static Light::SP create(Context *context,
                            const DevGroup::SP &devices,
                            const std::string &name);

    vec3f color = vec3f(1.f);
    DevGroup::SP const devices;
  };

};
