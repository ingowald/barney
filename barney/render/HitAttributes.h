// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/common/barney-common.h"

namespace BARNEY_NS {
  namespace render {
      
    enum { numAttributes = 4 };

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

    AttributeKind parseAttribute(const std::string &attributeName);
    
    struct HitAttributes {
      typedef AttributeKind Which;
        
      inline __rtc_device HitAttributes();
      inline __rtc_device vec4f get(Which attribute, bool dbg=false) const;
      
      vec4f color;
      vec4f attribute[numAttributes];
      vec3f worldPosition;
      vec3f objectPosition;
      vec3f worldNormal;
      vec3f objectNormal;
      int   primID;
      int   instID;
      float t;
      bool  isShadowRay = false;
    };

    inline __rtc_device HitAttributes::HitAttributes()
    {
      color
        // = vec4f(NAN,NAN,NAN,NAN);
        = vec4f(0.f,0.f,0.f,1.f);
      for (int i=0;i<numAttributes;i++)
        attribute[i]
          // = vec4f(NAN,NAN,NAN,NAN);
          = vec4f(0.f,0.f,0.f,1.f);
    }

    inline __rtc_device
    vec4f HitAttributes::get(Which whichOne, bool dbg) const
    {
      if (dbg) printf("HitAttributes::get(%i)\n",(int)whichOne);
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
      if (whichOne == WORLD_POSITION) {
        if (dbg) printf("worldpos \n");
        return vec4f(worldPosition.x,worldPosition.y,worldPosition.z,1.f);
        }
      if (whichOne == OBJECT_POSITION)
        return vec4f(objectPosition.x,objectPosition.y,objectPosition.z,1.f);
      if (dbg) printf("un-implemented hit attribute %i\n",(int)whichOne);
      return vec4f(0.f,0.f,0.f,1.f);
    }

  }
}
