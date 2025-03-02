// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

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
      vec4f baseColor = this->baseColor.eval(hitData,samplers,dbg);
      vec4f metallic = this->metallic.eval(hitData,samplers,dbg);
      vec4f opacity = this->opacity.eval(hitData,samplers,dbg);
      vec4f roughness = this->roughness.eval(hitData,samplers,dbg);
      vec4f transmission = this->transmission.eval(hitData,samplers,dbg);
      vec4f ior = this->ior.eval(hitData,samplers,dbg);
#if 1
      if (ior.x != 1.f && transmission.x >= 1e-3f) {
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
      bsdf.roughness = clamp(roughness.x,clampRange,1.f-clampRange);
      
      bsdf.alpha = (1.f-transmission.x)
        * baseColor.w
        * opacity.x
        ;
      
      bsdf.ior = ior.x;
      return bsdf;
    }
#endif
    
  }
}
