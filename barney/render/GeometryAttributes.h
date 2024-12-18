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

#include "barney/Object.h"
#include "barney/render/floatN.h"

namespace barney {
  namespace render {
      
    struct AttributeArray {
      struct DD {
        inline __device__ float4 valueAt(int i, bool dbg=false) const;
          
        const void       *ptr;
        int/*BNDataType*/ type;
        int               size;
      };
    };
      
    struct GeometryAttribute {
      typedef enum { INVALID, CONSTANT, PER_PRIM, PER_VERTEX  } Scope;
        
      struct DD {
        union {
          float4             value;
          AttributeArray::DD fromArray;
        };
        int/*Scope*/         scope;
      };
        
      vec4f       constant { 0.f,0.f,0.f,1.f };
      PODData::SP perPrim   = 0;
      PODData::SP perVertex = 0;
    };


    struct GeometryAttributes {
      enum { count = numAttributes };
      struct DD {
        enum { count = numAttributes };
        inline __device__ GeometryAttribute::DD &operator[](int i)
        { return attribute[i]; }
        inline __device__ const GeometryAttribute::DD &operator[](int i) const
        { return attribute[i]; }
        GeometryAttribute::DD attribute[numAttributes];
        GeometryAttribute::DD colorAttribute;
      };
      GeometryAttribute attribute[numAttributes];
      GeometryAttribute colorAttribute;
    };
  
      
    inline __device__
    float4 AttributeArray::DD::valueAt(int i, bool dbg) const
    {
      switch(this->type) {
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
      default:
        return make_float4(0.f,0.f,0.f,0.f);
      };
    }
    
  }
}
