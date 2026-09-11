// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/Object.h"
#include "native/render/floatN.h"
#include "native/common/Data.h"
#include "native/render/HitAttributes.h"
#include "native/render/TypedRead.h"

namespace BARNEY_NS {
  namespace native {
      
    struct AttributeArray {
      struct DD {
        inline __rtc_device vec4f valueAt(int i, bool dbg=false) const;
          
        const void       *ptr;
        int/*BNDataType*/ type;
        int               size;
      };
    };
      
    struct GeometryAttribute {
      typedef enum { INVALID, CONSTANT, PER_PRIM, PER_VERTEX, FACE_VARYING } Scope;
        
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
      PODData::SP perPrim     = 0;
      PODData::SP perVertex   = 0;
      PODData::SP faceVarying = 0;
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
        GeometryAttribute::DD normalAttribute;
      };
      DD getDD(Device *device);
      GeometryAttribute attribute[numAttributes];
      GeometryAttribute colorAttribute;
      GeometryAttribute normalAttribute;
    };
  
      
    inline __rtc_device
    vec4f AttributeArray::DD::valueAt(int i, bool dbg) const
    {
      return typedRead(ptr, type, i, size);
    }
    
  }
}
