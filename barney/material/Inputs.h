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

namespace barney {
  namespace render {

    struct Sampler {
      struct Globals {
        Sampler *samplers;
      };
    };
    
    struct HitAttributes {
      typedef enum {
        ATTRIBUTE_0,
        ATTRIBUTE_1,
        ATTRIBUTE_2,
        ATTRIBUTE_3,
        COLOR,
        WORLD_POSITION,
        WORLD_NORMAL,
        OBJECT_POSITION,
        OBJECT_NORMAL,
        PRIMITIVE_ID
      } Which;

      inline __device__ HitAttributes()
      {
        color = make_float4(0,0,0,1);
        for (int i=0;i<4;i++)
          attribute[i] = make_float4(0,0,0,1);
      }
      
      float4 color;
      float4 attribute[4];
      vec3f  worldPosition;
      vec3f  objectPosition;
      vec3f  worldNormal;
      vec3f  objectNormal;
      int    primID;
    };
    
    struct MaterialInput {
      typedef enum { VALUE, ATTRIBUTE, SAMPLER, UNDEFINED } Type;

      inline __device__
      float4 eval(const Sampler::Globals &samplers,
                  const HitAttributes &hitData) const;
      
      Type type;
      union {
        float4               value;
        HitAttributes::Which attribute;
        int                  samplerID;
      };
    };
    
  }
}
