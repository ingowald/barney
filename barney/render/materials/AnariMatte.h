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
      
    struct AnariMatte : public HostMaterial {      
      struct DD {
        inline __device__
        PackedBSDF createBSDF(const HitAttributes &hitData, bool dbg) const;
        PossiblyMappedParameter::DD color;
      };
      AnariMatte(ModelSlot *owner) : HostMaterial(owner) {}
      virtual ~AnariMatte() = default;

      std::string toString() const override { return "AnariMatte"; }
      
      void createDD(DeviceMaterial &dd, int deviceID) const override
      { throw std::runtime_error("AnriMatte::createDD( not implemented..."); }
      
      PossiblyMappedParameter color;
    };
      
    inline __device__
    PackedBSDF AnariMatte::DD::createBSDF(const HitAttributes &hitData, bool dbg) const
    {
      float4 r = color.eval(hitData);
        
      return packedBSDF::VisRTX::make_matte((const vec3f&)r);
    }
      
  }
}