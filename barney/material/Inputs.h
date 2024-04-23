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
    enum { numAttributes = 4 };

    struct DeviceMaterial;
    
    struct Sampler {
      struct DD {
      };
    };

    struct MaterialGlobals {
      render::DeviceMaterial *materials;
      render::Sampler        *samplers;
    };

    struct GeometryAttribute {
      
      typedef enum { PER_VERTEX, PER_PRIM, CONSTANT, INVALID } Scope;
      
      struct DataArray {
        const void      *ptr;
        int/*BNDataType*/type;
        
        inline __device__
        float4 valueAt(int i) const;
      };
      
      struct DD {
        union {
          float4    value;
          DataArray fromArray;
        };
        int/*Scope*/     scope;
      };
      struct OnHost {
        vec4f       constant { 0.f,0.f,0.f,1.f };
        PODData::SP perPrim   = 0;
        PODData::SP perVertex = 0;
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

      inline __device__ HitAttributes(const MaterialGlobals &globals)
        : globals(globals)
      {
        color = make_float4(0,0,0,1);
        for (int i=0;i<numAttributes;i++)
          attribute[i] = make_float4(0,0,0,1);
      }

      inline __device__ float4 get(Which attribute) const;
      
      float4 color;
      float4 attribute[numAttributes];
      vec3f  worldPosition;
      vec3f  objectPosition;
      vec3f  worldNormal;
      vec3f  objectNormal;
      int    primID;
      float  t;

      const MaterialGlobals &globals;
    };

    struct MaterialInput {
      typedef enum { VALUE, ATTRIBUTE, SAMPLER, UNDEFINED } Type;
      
      inline __device__
      float4 eval(const HitAttributes &hitData) const;
      
      Type type;
      union {
        float4               value;
        HitAttributes::Which attribute;
        int                  samplerID;
      };
    };

    inline __device__
    float4 HitAttributes::get(Which whichOne) const
    {
      if (whichOne == ATTRIBUTE_0)
        return attribute[0];
      
      printf("un-handled hit-data attribute %i\n",int(whichOne));
      return make_float4(0.f,0.f,0.f,1.f);
    }
    
    inline __device__
    float4 MaterialInput::eval(const HitAttributes &hitData) const
    {
      if (type == VALUE)
        return value;
      if (type == ATTRIBUTE)
        return hitData.get(attribute);
      printf("un-handled material input type\n");
      return make_float4(0.f,0.f,0.f,1.f);
    }
      
    inline __device__
    float4 GeometryAttribute::DataArray::valueAt(int i) const
    {
      switch(type) {
      case BN_FLOAT: {
        const float v = ((const float *)ptr)[i];
        return make_float4(v,0.f,0.f,1.f);
      }
      case BN_FLOAT2: {
        const float2 v = ((const float2 *)ptr)[i];
        return make_float4(v.x,v.y,0.f,1.f);
      }
      case BN_FLOAT3: {
        const float3 v = ((const float3 *)ptr)[i];
        return make_float4(v.x,v.y,v.z,1.f);
      }
      case BN_FLOAT4: {
        const float4 v = ((const float4 *)ptr)[i];
        return v;
      }
      };
    }
    
  }
}
