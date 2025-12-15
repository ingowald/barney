// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/common/barney-common.h"
#include "barney/packedBSDF/PackedBSDF.h"
#include "barney/packedBSDF/NVisii.h"
#include "barney/render/HitAttributes.h"
#include "barney/material/Material.h"

namespace BARNEY_NS {
  namespace render {
      
    struct AnariMatte : public HostMaterial {      
      struct DD {
#if RTC_DEVICE_CODE
        inline __rtc_device
        PackedBSDF createBSDF(const HitAttributes &hitData,
                              const Sampler::DD *samplers,
                              bool dbg) const;
#endif
        PossiblyMappedParameter::DD color;
        PossiblyMappedParameter::DD opacity;
      };
      AnariMatte(SlotContext *context) : HostMaterial(context) {}
      virtual ~AnariMatte() = default;

      bool setString(const std::string &member,
                     const std::string &value) override;
      bool set1f(const std::string &member,
                 const float &value) override;
      bool set3f(const std::string &member,
                 const vec3f &value) override;
      bool set4f(const std::string &member,
                 const vec4f &value) override;
      bool setObject(const std::string &member,
                     const Object::SP &value) override;
      
      std::string toString() const override { return "AnariMatte"; }
      
      DeviceMaterial getDD(Device *device) override;
      
      PossiblyMappedParameter color = vec3f(.8f);
    };
      
#if RTC_DEVICE_CODE
    inline __rtc_device
    PackedBSDF AnariMatte::DD::createBSDF(const HitAttributes &hitData,
                                          const Sampler::DD *samplers,
                                          bool dbg) const
    {
      if (dbg) printf("anarimatte createbsdf\n");
      vec4f baseColor = this->color.eval(hitData,samplers,dbg);
      vec4f opacity = this->opacity.eval(hitData,samplers,dbg);
# if 1
      float reflectance = .85f;
      packedBSDF::Lambertian bsdf;
      (vec3f&)bsdf.albedo = reflectance * (const vec3f&)baseColor
        * (ONE_OVER_PI)
        ;
      if (dbg) printf("created lambertian %f %f %f\n",
                      bsdf.albedo.x,
                      bsdf.albedo.y,
                      bsdf.albedo.z);
      bsdf.alpha = baseColor.w * opacity.x;
# else
      packedBSDF::NVisii bsdf;
      bsdf.setDefaults();

      bsdf.baseColor = (const vec3f&)baseColor;

      bsdf.specular = 0.f;
      bsdf.metallic = 0.0f;
      bsdf.roughness = 1.f;
      bsdf.ior = 1.f;
      // bsdf.alpha = baseColor.w;
# endif
      return bsdf;
    }
#endif    
  }
}
