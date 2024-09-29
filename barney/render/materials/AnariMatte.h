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
#include "barney/render/packedBSDFs/NVisii.h"
#include "barney/render/HitAttributes.h"
#include "barney/render/HostMaterial.h"

namespace barney {
  namespace render {
      
    struct AnariMatte : public HostMaterial {      
      struct DD {
#ifdef __CUDA_ARCH__
        inline __device__
        PackedBSDF createBSDF(const HitAttributes &hitData,
                              const Sampler::DD *samplers,
                              bool dbg) const;
#endif
        PossiblyMappedParameter::DD color;
      };
      AnariMatte(ModelSlot *owner) : HostMaterial(owner) {}
      virtual ~AnariMatte() = default;

      bool setString(const std::string &member, const std::string &value) override;
      bool set3f(const std::string &member, const vec3f &value) override;
      bool setObject(const std::string &member, const Object::SP &value) override;
      
      std::string toString() const override { return "AnariMatte"; }
      
      void createDD(DeviceMaterial &dd, int deviceID) const override;
      
      PossiblyMappedParameter color = vec3f(.8f);
    };
      
  
#ifdef __CUDA_ARCH__
    inline __device__
    PackedBSDF AnariMatte::DD::createBSDF(const HitAttributes &hitData,
                                          const Sampler::DD *samplers,
                                          bool dbg) const
    {
      packedBSDF::NVisii bsdf;
      bsdf.setDefaults();
      
      float4 baseColor = this->color.eval(hitData,samplers,dbg);

      // not anari-conformant, but useful: if geometry _has_ a color
      // attribute, use it, no matter whether our input is point to it
      // or not:
      // if (!isnan(hitData.color.x)) 
      //   baseColor = hitData.color;

      bsdf.baseColor = (const vec3f&)baseColor;

      bsdf.specular = 0.f;
      bsdf.metallic = 0.f;
      bsdf.roughness = .5f;
      return bsdf;
    }
#endif
    
  }
}
