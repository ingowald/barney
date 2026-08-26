// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/packedBSDF/PackedBSDF.h"
#include "native/packedBSDF/NVisii.h"
#include "native/material/HostMaterial.h"

namespace BARNEY_NS {
  namespace native {

    /*! Disney "Principled" BSDF material, exposed through the barney
        core as the "nvisii" subtype. Wraps the packedBSDF::NVisii
        BSDF (a port of the NVIDIA NVisii Disney material), carrying
        the full Disney parameter set. All scalar parameters are
        stored as half precision in the device BSDF, matching the
        packedBSDF::NVisii struct. */
    struct NVisii : public HostMaterial {
      struct DD {
#if RTC_DEVICE_CODE
        inline __rtc_device
        PackedBSDF createBSDF(const HitAttributes &hitData,
                              const Sampler::DD *samplers,
                              bool dbg) const;
#endif
        PossiblyMappedParameter::DD baseColor;
        PossiblyMappedParameter::DD subsurfaceColor;
        PossiblyMappedParameter::DD metallic;
        PossiblyMappedParameter::DD specular;
        PossiblyMappedParameter::DD roughness;
        PossiblyMappedParameter::DD specularTint;
        PossiblyMappedParameter::DD anisotropy;
        PossiblyMappedParameter::DD sheen;
        PossiblyMappedParameter::DD sheenTint;
        PossiblyMappedParameter::DD clearcoat;
        PossiblyMappedParameter::DD clearcoatGloss;
        PossiblyMappedParameter::DD ior;
        PossiblyMappedParameter::DD specularTransmission;
        PossiblyMappedParameter::DD transmissionRoughness;
        PossiblyMappedParameter::DD flatness;
        PossiblyMappedParameter::DD opacity;
      };

      NVisii(LDGContext *context);
      virtual ~NVisii() = default;

      std::string toString() const override { return "NVisii"; }

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

      // Defaults mirror packedBSDF::NVisii::setDefaults().
      PossiblyMappedParameter baseColor            = vec3f(0.8f);
      PossiblyMappedParameter subsurfaceColor      = vec3f(0.8f);
      PossiblyMappedParameter metallic             = 0.f;
      PossiblyMappedParameter specular             = 0.5f;
      PossiblyMappedParameter roughness            = 0.5f;
      PossiblyMappedParameter specularTint         = 0.f;
      PossiblyMappedParameter anisotropy           = 0.f;
      PossiblyMappedParameter sheen                = 0.f;
      PossiblyMappedParameter sheenTint            = 0.5f;
      PossiblyMappedParameter clearcoat            = 0.f;
      // setDefaults: clearcoat_roughness = 0.03 -> gloss = 1 - 0.03^2.
      PossiblyMappedParameter clearcoatGloss      = 1.f - 0.03f * 0.03f;
      PossiblyMappedParameter ior                  = 1.45f;
      PossiblyMappedParameter specularTransmission = 0.f;
      PossiblyMappedParameter transmissionRoughness = 0.04f;
      PossiblyMappedParameter flatness             = 0.f;
      PossiblyMappedParameter opacity              = 1.f;
    };

#if RTC_DEVICE_CODE
    inline __rtc_device
    PackedBSDF NVisii::DD::createBSDF(const HitAttributes &hitData,
                                      const Sampler::DD *samplers,
                                      bool dbg) const
    {
      vec4f baseColor            = this->baseColor           .eval(hitData,samplers,dbg);
      vec4f subsurfaceColor      = this->subsurfaceColor     .eval(hitData,samplers,dbg);
      vec4f metallic             = this->metallic            .eval(hitData,samplers,dbg);
      vec4f specular             = this->specular            .eval(hitData,samplers,dbg);
      vec4f roughness            = this->roughness           .eval(hitData,samplers,dbg);
      vec4f specularTint         = this->specularTint        .eval(hitData,samplers,dbg);
      vec4f anisotropy           = this->anisotropy          .eval(hitData,samplers,dbg);
      vec4f sheen                = this->sheen               .eval(hitData,samplers,dbg);
      vec4f sheenTint            = this->sheenTint           .eval(hitData,samplers,dbg);
      vec4f clearcoat            = this->clearcoat           .eval(hitData,samplers,dbg);
      vec4f clearcoatGloss       = this->clearcoatGloss      .eval(hitData,samplers,dbg);
      vec4f ior                  = this->ior                 .eval(hitData,samplers,dbg);
      vec4f specularTransmission = this->specularTransmission.eval(hitData,samplers,dbg);
      vec4f transmissionRoughness= this->transmissionRoughness.eval(hitData,samplers,dbg);
      vec4f flatness             = this->flatness            .eval(hitData,samplers,dbg);
      vec4f opacity              = this->opacity             .eval(hitData,samplers,dbg);

      packedBSDF::NVisii bsdf;
      bsdf.setDefaults();
      bsdf.baseColor            = (const vec3f&)baseColor;
      bsdf.subsurfaceColor      = (const vec3f&)subsurfaceColor;
      bsdf.metallic             = metallic.x;
      bsdf.specular             = specular.x;
      bsdf.roughness            = roughness.x;
      bsdf.specularTint         = specularTint.x;
      bsdf.anisotropy           = anisotropy.x;
      bsdf.sheen                = sheen.x;
      bsdf.sheenTint            = sheenTint.x;
      bsdf.clearcoat            = clearcoat.x;
      bsdf.clearcoatGloss       = clearcoatGloss.x;
      bsdf.ior                  = ior.x;
      bsdf.specularTransmission = specularTransmission.x;
      bsdf.transmissionRoughness= transmissionRoughness.x;
      bsdf.flatness             = flatness.x;
      bsdf.alpha                = opacity.x;
      return bsdf;
    }
#endif

  }
}
