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

#include "barney/render/device/HitAttributes.h"

namespace BARNEY_NS {
  namespace render {
    namespace device {

      xxxx
      
      struct MaterialInput {
        typedef enum { VALUE, ATTRIBUTE, SAMPLER, UNDEFINED } Type;
      
        inline __rtc_device
        float4 eval(const HitAttributes &hitData) const;
      
        Type type;
        union {
          float4               value;
          HitAttributes::Which attribute;
          int                  samplerID;
        };
      };

      inline __rtc_device
      float4 MaterialInput::eval(const HitAttributes &hitData,
                                 Sampler::DD *samplers) const
      {
        if (type == VALUE)
          return value;
        if (type == ATTRIBUTE)
          return hitData.get(attribute);
        if (type == SAMPLER) { 
          if (samplerID < 0) return make_float4(0.f,0.f,0.f,1.f);
          return samplers[samplerID].eval(hitData);
        }
        printf("un-handled material input type\n");
        return make_float4(0.f,0.f,0.f,1.f);
      }

    }
  }
}
