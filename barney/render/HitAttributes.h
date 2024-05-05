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

#include "barney/DeviceContext.h"

namespace barney {
  namespace render {
      
    enum { numAttributes = 4 };

    struct Material;
    struct Sampler;
      
    typedef enum {
      ATTRIBUTE_KIND_NONE=0,
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
    } AttributeKind;
    
    struct HitAttributes {
      // struct Globals {
      //   const Material *materials;
      //   const Sampler  *samplers;
      // };
        
      typedef AttributeKind Which;
        
      inline __device__ HitAttributes(// const Globals &globals
                                        )
      // : globals(globals)
      {
        color = make_float4(0,0,0,1);
        for (int i=0;i<numAttributes;i++)
          attribute[i] = make_float4(0,0,0,1);
      }

      inline __device__ float4 get(Which attribute, bool dbg=false) const;
      
      float4 color;
      float4 attribute[numAttributes];
      vec3f  worldPosition;
      vec3f  objectPosition;
      vec3f  worldNormal;
      vec3f  objectNormal;
      int    primID;
      float  t;

      // const Globals &globals;
    };

    inline __device__
    float4 HitAttributes::get(Which whichOne, bool dbg) const
    {
      // if (dbg)
      //   printf("hit attribute which %i\n",int(whichOne));
      
      if (whichOne == ATTRIBUTE_0)
        return attribute[0];
      if (whichOne == ATTRIBUTE_1)
        return attribute[1];
      if (whichOne == ATTRIBUTE_2)
        return attribute[2];
      if (whichOne == ATTRIBUTE_3)
        return attribute[3];
      if (whichOne == COLOR)
        return color;
      
      printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% un-handled hit-data attribute %i\n",int(whichOne));
      return make_float4(0.f,0.f,0.f,1.f);
    }

  }
}
