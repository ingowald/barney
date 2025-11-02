// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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
