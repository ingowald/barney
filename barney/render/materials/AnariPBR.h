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
      };
      AnariPBR(ModelSlot *owner) : HostMaterial(owner) {}
      virtual ~AnariPBR() = default;
      
      std::string toString() const override { return "AnariPBR"; }
      
      void createDD(DeviceMaterial &dd, int deviceID) const override;

      bool setString(const std::string &member, const std::string &value) override;
      bool set1f(const std::string &member, const float &value) override;
      bool set3f(const std::string &member, const vec3f &value) override;
      
      PossiblyMappedParameter baseColor;
      PossiblyMappedParameter metallic;
      PossiblyMappedParameter roughness;
    };
      
    inline __device__
    PackedBSDF AnariPBR::DD::createBSDF(const HitAttributes &hitData, bool dbg) const
    {
      packedBSDF::VisRTX bsdf;
      
      float4 baseColor = this->baseColor.eval(hitData);
      bsdf.baseColor = (const vec3f&)baseColor;

      float4 metallic = this->metallic.eval(hitData,dbg);
      bsdf.metallic = metallic.x;

      float4 roughness = this->roughness.eval(hitData,dbg);
      if (dbg) printf("got roughness %f %f %f %f\n",
                      roughness.x,
                      roughness.y,
                      roughness.z,
                      roughness.w);
      bsdf.roughness = roughness.x;
      
      bsdf.ior = 1.5f;
      bsdf.opacity = 1.f;

      return bsdf;
    }

  }
}
