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

namespace barney {
  namespace render {
    
    struct AnariPBR : public HostMaterial {
      struct DD {
       inline __device__
        PackedBSDF createBSDF(const HitAttributes &hitData,
                              const Sampler::DD *samplers,
                              bool dbg) const;
        inline __device__
        float getOpacity(const HitAttributes &hitData,
                         const Sampler::DD *samplers,
                         bool dbg) const;
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

      bool setObject(const std::string &member, const Object::SP &value) override;
      bool setString(const std::string &member, const std::string &value) override;
      bool set1f(const std::string &member, const float &value) override;
      bool set3f(const std::string &member, const vec3f &value) override;
      bool set4f(const std::string &member, const vec4f &value) override;
      
      PossiblyMappedParameter baseColor    = vec3f(1.f,1.f,1.f);
      PossiblyMappedParameter metallic     = 1.f;
      PossiblyMappedParameter roughness    = 1.f;
      PossiblyMappedParameter transmission = 0.f;
      PossiblyMappedParameter ior          = 1.45f;
      PossiblyMappedParameter emission     = vec3f(0.f,0.f,0.f);
    };
      
#ifdef __CUDACC__
    inline __device__
    PackedBSDF AnariPBR::DD::createBSDF(const HitAttributes &hitData,
                                        const Sampler::DD *samplers,
                                        bool dbg) const
    {
      float4 baseColor = this->baseColor.eval(hitData,samplers,dbg);
      float4 metallic = this->metallic.eval(hitData,samplers,dbg);
      float4 roughness = this->roughness.eval(hitData,samplers,dbg);
      float4 transmission = this->transmission.eval(hitData,samplers,dbg);
      float4 ior = this->ior.eval(hitData,samplers,dbg);
#if 1
      // if (dbg) printf("ior %f trans %f\n",ior.x,transmission.x);
      if (ior.x != 1.f && transmission.x >= 1e-3f) {
        packedBSDF::Glass bsdf;
        bsdf.ior = ior.x;
        bsdf.attenuation = vec3f(1.f);
        // if (dbg) printf("MADE GLASS\n");
        // bsdf.ior = 1.45f;
        // bsdf.specularTransmission = 1.f;
        // bsdf.baseColor = vec3f(1.f);
        // bsdf.metallic = 0.f;
        // bsdf.roughness = 0.f;
        // bsdf.specular = 0.f;
        return bsdf;
      }
#endif
      packedBSDF::NVisii bsdf;
      bsdf.setDefaults();
      const float clampRange = .05f;
      
      bsdf.baseColor = (const vec3f&)baseColor;
      bsdf.metallic = clamp(metallic.x,clampRange,1.f-clampRange);
      bsdf.roughness = clamp(roughness.x,clampRange,1.f-clampRange);
      
      bsdf.alpha = (1.f-transmission.x)
        * baseColor.w
        ;
      
      bsdf.ior = ior.x;
      // if (dbg)
      //   printf("created nvisii brdf, base %f %f %f metallic %f roughness %f ior %f alpha %f\n",
      //          (float)bsdf.baseColor.x,
      //          (float)bsdf.baseColor.y,
      //          (float)bsdf.baseColor.z,
      //          (float)bsdf.metallic,
      //          (float)bsdf.roughness,
      //          (float)bsdf.ior,
      //                 (float)bsdf.alpha);
      return bsdf;
    }



    inline __device__
    float AnariPBR::DD::getOpacity(const HitAttributes &hitData,
                                   const Sampler::DD *samplers,
                                   bool dbg) const
    {
      float4 baseColor = this->baseColor.eval(hitData,samplers,dbg);
      return baseColor.w;
    }
#endif
    
  }
}
