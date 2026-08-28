// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/packedBSDF/PackedBSDF.h"
#include "native/packedBSDF/Glass.h"
#include "native/material/HostMaterial.h"

namespace BARNEY_NS {
  namespace native {

    /*! smooth dielectric material, exposed through the barney core as the
        "glass" subtype. Wraps the packedBSDF::Glass BSDF (specular
        reflect/refract through the robust OSPRay dielectric), carrying an
        index of refraction and a Beer-Lambert attenuation color. Glass is
        transparent to shadow rays; its transmission is handled by scatter,
        so coverage stays fully opaque. */
    struct Glass : public HostMaterial {
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
        PossiblyMappedParameter::DD ior;
        PossiblyMappedParameter::DD attenuationColor;
      };

      Glass(LDGContext *context);
      virtual ~Glass() = default;

      std::string toString() const override { return "Glass"; }

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

      PossiblyMappedParameter ior              = 1.45f;
      PossiblyMappedParameter attenuationColor = vec3f(1.f,1.f,1.f);
    };

#if RTC_DEVICE_CODE
    inline __rtc_device
    PackedBSDF Glass::DD::createBSDF(const HitAttributes &hitData,
                                     const Sampler::DD *samplers,
                                     bool dbg) const
    {
      vec4f ior              = this->ior             .eval(hitData,samplers,dbg);
      vec4f attenuationColor = this->attenuationColor.eval(hitData,samplers,dbg);
      packedBSDF::Glass bsdf;
      bsdf.ior = fmaxf(ior.x, 1e-3f);
      bsdf.attenuation
        = rtc::float3(attenuationColor.x,attenuationColor.y,attenuationColor.z);
      return bsdf;
    }

    inline __rtc_device
    float Glass::DD::getOpacity(const HitAttributes &hitData,
                               const Sampler::DD *samplers,
                               bool dbg) const
    {
      // Glass is a smooth dielectric: transmission is resolved in scatter(),
      // not through cut-out coverage, so it always presents as fully opaque
      // to the coverage/alphaMode path.
      return 1.f;
    }
#endif

  }
}
