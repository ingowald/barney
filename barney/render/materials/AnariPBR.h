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

#include "barney/render/packedBSDFs/VisRTX.h"
#include "barney/render/HitAttributes.h"
#include "barney/render/HostMaterial.h"

namespace barney {
  namespace render {
    
    struct AnariPBR : public HostMaterial {
      struct DD {
        inline __device__
        PackedBSDF createBSDF(const HitAttributes &hitData, bool dbg) const;
        PossiblyMappedParameter::DD baseColor;
        PossiblyMappedParameter::DD metallic;
        PossiblyMappedParameter::DD roughness;
        PossiblyMappedParameter::DD transmission;
        PossiblyMappedParameter::DD ior;
        PossiblyMappedParameter::DD emission;
      };
      AnariPBR(ModelSlot *owner) : HostMaterial(owner) {}
      virtual ~AnariPBR() = default;
      
      std::string toString() const override { return "AnariPBR"; }
      
      void createDD(DeviceMaterial &dd, int deviceID) const override;

      bool setString(const std::string &member, const std::string &value) override;
      bool set1f(const std::string &member, const float &value) override;
      bool set3f(const std::string &member, const vec3f &value) override;
      
      PossiblyMappedParameter baseColor    = vec3f(1.f,1.f,1.f);
      PossiblyMappedParameter metallic     = 1.f;
      PossiblyMappedParameter roughness    = 1.f;
      PossiblyMappedParameter transmission = 0.f;
      PossiblyMappedParameter ior          = 1.45f;
      PossiblyMappedParameter emission     = vec3f(0.f,0.f,0.f);
    };
      
    inline __device__
    PackedBSDF AnariPBR::DD::createBSDF(const HitAttributes &hitData, bool dbg) const
    {
      const float clampRange = .01f;
#if 1
      packedBSDF::NVisii bsdf;
      bsdf.setDefaults();
      float4 baseColor = this->baseColor.eval(hitData);
      bsdf.baseColor = (const vec3f&)baseColor;
      float4 metallic = this->metallic.eval(hitData,dbg);
      bsdf.metallic = clamp(metallic.x,clampRange,1.f-clampRange);
      float4 roughness = this->roughness.eval(hitData,dbg);
      bsdf.roughness = clamp(roughness.x,clampRange,1.f-clampRange);
      
      float4 transmission = this->transmission.eval(hitData,dbg);
      bsdf.alpha = 1.f-transmission.x;
      
      float4 ior = this->ior.eval(hitData,dbg);
      bsdf.ior = ior.x;
      if (dbg) printf("created nvisii brdf, base %f %f %f\n",
                      (float)bsdf.baseColor.x,
                      (float)bsdf.baseColor.y,
                      (float)bsdf.baseColor.z);
#else
      packedBSDF::VisRTX bsdf;
      
      float4 baseColor = this->baseColor.eval(hitData);
      bsdf.baseColor = (const vec3f&)baseColor;

      float4 metallic = this->metallic.eval(hitData,dbg);
      bsdf.metallic = clamp(metallic.x,clampRange,1.f-clampRange);

      float4 roughness = this->roughness.eval(hitData,dbg);
      bsdf.roughness = clamp(roughness.x,clampRange,1.f-clampRange);
      
      float4 transmission = this->transmission.eval(hitData,dbg);
      bsdf.opacity = 1.f-transmission.x;
      
      float4 ior = this->ior.eval(hitData,dbg);
      bsdf.ior = ior.x;

      if (dbg)
        printf("### AnariPBR created BSDF baseColor %f %f %f metallic %f roughness %f ior %f\n",
               baseColor.x,
               baseColor.y,
               baseColor.z,
               (float)bsdf.metallic,
               (float)bsdf.roughness,
               (float)bsdf.ior);
#endif
      return bsdf;
    }

  }
}
