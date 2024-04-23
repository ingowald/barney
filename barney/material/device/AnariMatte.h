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
#include "barney/material/Inputs.h"

namespace barney {
  namespace render {
    namespace device {
      
      struct AnariMatte {
        inline __device__
        PackedBSDF createBSDF(const Sampler::Globals &samplers,
                              const HitAttributes &hitData) const;
        
        MaterialInput reflectance;
      };
      
      inline __device__
      PackedBSDF AnariMatte::createBSDF(const Sampler::Globals &samplers,
                                        const HitAttributes &hitData) const
      {
        float4 r = reflectance.eval(samplers,hitData);
        
        return packedBSDF::VisRTX::make_matte((const vec3f&)r);
      }
      
    }
  }
}
