// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/packedBSDF/PackedBSDF.h"
#include "barney/material/Material.h"

namespace BARNEY_NS {
  namespace render {
    
    struct AnariPBR : public HostMaterial {
      struct DD {
#if RTC_DEVICE_CODE
       inline __rtc_device
        PackedBSDF createBSDF(const HitAttributes &hitData,
                              const Sampler::DD *samplers,
                              bool dbg) const;
        inline __rtc_device
        float getOpacity(const HitAttributes &hitData,
                         const Sampler::DD *samplers,
                         bool dbg) const;
#endif
        PossiblyMappedParameter::DD baseColor;
        PossiblyMappedParameter::DD metallic;
        PossiblyMappedParameter::DD opacity;
        PossiblyMappedParameter::DD roughness;
        PossiblyMappedParameter::DD transmission;
        PossiblyMappedParameter::DD ior;
        PossiblyMappedParameter::DD emission;
      };
      
      AnariPBR(SlotContext *context);
      virtual ~AnariPBR() = default;
      
      std::string toString() const override { return "AnariPBR"; }
      
      DeviceMaterial getDD(Device *device) override;

      bool setObject(const std::string &member,
                     const Object::SP &value) override;
      bool setString(const std::string &member,
                     const std::string &value) override;
      bool set1f(const std::string &member,
                 const float &value) override;
      bool set3f(const std::string &member,
                 const vec3f &value) override;
      bool set4f(const std::string &member,
                 const vec4f &value) override;
      
      PossiblyMappedParameter baseColor    = vec3f(1.f,1.f,1.f);
      PossiblyMappedParameter metallic     = 1.f;
      PossiblyMappedParameter opacity      = 1.f;
      PossiblyMappedParameter roughness    = 1.f;
      PossiblyMappedParameter transmission = 0.f;
      PossiblyMappedParameter ior          = 1.45f;
      PossiblyMappedParameter emission     = vec3f(0.f,0.f,0.f);
    };
      
#if RTC_DEVICE_CODE
    inline __rtc_device
    PackedBSDF AnariPBR::DD::createBSDF(const HitAttributes &hitData,
                                        const Sampler::DD *samplers,
                                        bool dbg) const
    {
      if (dbg) printf("createBSDF\n");
      vec4f baseColor = this->baseColor.eval(hitData,samplers,dbg);
      if (dbg) printf("basecolor eval %f %f %f %f\n",
                      baseColor.x,
                      baseColor.y,
                      baseColor.z,
                      baseColor.w);
      if (dbg) printf("evaling metallic\n");
      vec4f metallic = this->metallic.eval(hitData,samplers,dbg);
      vec4f opacity = this->opacity.eval(hitData,samplers,dbg);
      vec4f roughness = this->roughness.eval(hitData,samplers,dbg);
      vec4f transmission = this->transmission.eval(hitData,samplers,dbg);
      vec4f ior = this->ior.eval(hitData,samplers,dbg);
#if 1
      if (ior.x != 1.f && (transmission.x >= 1e-3f// || opacity.x < 1.f
                           )) {
        packedBSDF::Glass bsdf;
        bsdf.ior = ior.x;
        (vec3f&)bsdf.attenuation = vec3f(1.f);
        return bsdf;
      }
#endif
      packedBSDF::NVisii bsdf;
      bsdf.setDefaults();
      const float clampRange = .1f;
      bsdf.baseColor = (const vec3f&)baseColor;
      bsdf.metallic = metallic.x;
      bsdf.roughness = clamp(roughness.x,.001f,1.f-clampRange);

      bsdf.alpha = (1.f-transmission.x)
        * baseColor.w
        * opacity.x
        ;
      
      bsdf.ior = ior.x;
      if (dbg) printf("created nvisii bsdf base %f %f %f met %f base rough %f ior %f alpha %f\n",
                      (float)bsdf.baseColor.x,
                      (float)bsdf.baseColor.y,
                      (float)bsdf.baseColor.z,
                      (float)bsdf.metallic,
                      (float)bsdf.roughness,
                      (float)bsdf.ior,
                      (float)bsdf.alpha);
      return bsdf;
    }
#endif
    
  }
}
