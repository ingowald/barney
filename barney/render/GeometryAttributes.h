// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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
#include "barney/common/Data.h"
#include "barney/render/HitAttributes.h"

namespace BARNEY_NS {
  namespace render {
      
    struct AttributeArray {
      struct DD {
        inline __rtc_device vec4f valueAt(int i, bool dbg=false) const;
          
        const void       *ptr;
        int/*BNDataType*/ type;
        int               size;
      };
    };
      
    struct GeometryAttribute {
      typedef enum { INVALID, CONSTANT, PER_PRIM, PER_VERTEX  } Scope;
        
      struct DD {
        // union {
          /*! careful - this requires alignment, which means that the
              follwing value - even if just a int - will also require
              16 bytes, every time */
          vec4f        value;
          AttributeArray::DD fromArray;
        // };
        int/*Scope*/         scope;
      };

      /*! initialize host-side 'constant' value to 'NAN' to indicate
          that they have not (yet) been set. This allows to then set
          the device-side scopt to 'invalid' for such values */
      vec4f       constant { NAN,NAN,NAN,NAN };
      PODData::SP perPrim   = 0;
      PODData::SP perVertex = 0;
    };


    struct GeometryAttributes {
      enum { count = numAttributes };
      struct DD {
        enum { count = numAttributes };
        inline __rtc_device GeometryAttribute::DD &operator[](int i)
        { return attribute[i]; }
        inline __rtc_device const GeometryAttribute::DD &operator[](int i) const
        { return attribute[i]; }
        GeometryAttribute::DD attribute[numAttributes];
        GeometryAttribute::DD colorAttribute;
      };
      DD getDD(Device *device);
      GeometryAttribute attribute[numAttributes];
      GeometryAttribute colorAttribute;
    };
  
      
    inline __rtc_device
    vec4f AttributeArray::DD::valueAt(int i, bool dbg) const
    {
      switch(this->type) {
      case BN_FLOAT: {
        const float v = ((const float *)ptr)[i];
        return vec4f(v,0.f,0.f,1.f);
      }
      case BN_FLOAT2: {
        const vec2f v = ((const vec2f *)ptr)[i];
        return vec4f(v.x,v.y,0.f,1.f);
      }
      case BN_FLOAT3: {
        const vec3f v = ((const vec3f *)ptr)[i];
        return vec4f(v.x,v.y,v.z,1.f);
      }
      case BN_FLOAT4: {
        return ((const vec4f *)ptr)[i];
        // return rtc::load(((const rtc::float4*)ptr)[i]);
      }
      default:
        return vec4f(0.f,0.f,0.f,0.f);
      };
    }
    
  }
}
