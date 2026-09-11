// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The single device-side read-time expansion point: reads a vec4f from a
// typed array at an index, with decode formulas matching ANARI's
// toFloat4 exactly (missing components 0, alpha 1, normalized integers
// divided by their max). Shared by the geometry attribute reads and the
// primitive sampler so decode semantics cannot drift between consumers.
// The includer must provide the vec types and the BNDataType constants.

namespace BARNEY_NS {
  namespace native {

    // size < 0 means "unknown length" (no bounds check). Out-of-range
    // indices return (0,0,0,0) - same as an unrecognized type.
    inline __rtc_device
    vec4f typedRead(const void *ptr, int /*BNDataType*/ type, int i,
                    int size = -1)
    {
      if (!ptr || i < 0 || (size >= 0 && i >= size))
        return vec4f(0.f, 0.f, 0.f, 0.f);
      switch (type) {
      case BN_FLOAT32: {
        const float v = ((const float *)ptr)[i];
        return vec4f(v, 0.f, 0.f, 1.f);
      }
      case BN_FLOAT32_VEC2: {
        const vec2f v = ((const vec2f *)ptr)[i];
        return vec4f(v.x, v.y, 0.f, 1.f);
      }
      case BN_FLOAT32_VEC3: {
        const vec3f v = ((const vec3f *)ptr)[i];
        return vec4f(v.x, v.y, v.z, 1.f);
      }
      case BN_FLOAT32_VEC4: {
        return ((const vec4f *)ptr)[i];
      }
      case BN_UFIXED8: {
        const float v = (float)((const uint8_t *)ptr)[i];
        return vec4f(v * (1.f/255.f), 0.f, 0.f, 1.f);
      }
      case BN_UFIXED8_VEC2: {
        const vec2uc v = ((const vec2uc *)ptr)[i];
        return vec4f(v.x * (1.f/255.f), v.y * (1.f/255.f), 0.f, 1.f);
      }
      case BN_UFIXED8_RGBA: {
        const vec4uc v = ((const vec4uc *)ptr)[i];
        return vec4f(v) * (1.f/255.f);
      }
      case BN_UFIXED16: {
        const float v = (float)((const uint16_t *)ptr)[i];
        return vec4f(v * (1.f/65535.f), 0.f, 0.f, 1.f);
      }
      case BN_UFIXED16_VEC2: {
        const vec2us v = ((const vec2us *)ptr)[i];
        return vec4f(v.x * (1.f/65535.f), v.y * (1.f/65535.f), 0.f, 1.f);
      }
      case BN_FLOAT16: {
        const uint16_t v = ((const uint16_t *)ptr)[i];
        return vec4f(rtc::halfToFloat(v), 0.f, 0.f, 1.f);
      }
      case BN_FLOAT16_VEC2: {
        const uint16_t *v = &((const uint16_t *)ptr)[i*2];
        return vec4f(rtc::halfToFloat(v[0]), rtc::halfToFloat(v[1]),
                     0.f, 1.f);
      }
      case BN_FLOAT16_VEC3: {
        const uint16_t *v = &((const uint16_t *)ptr)[i*3];
        return vec4f(rtc::halfToFloat(v[0]), rtc::halfToFloat(v[1]),
                     rtc::halfToFloat(v[2]), 1.f);
      }
      case BN_FLOAT16_VEC4: {
        const uint16_t *v = &((const uint16_t *)ptr)[i*4];
        return vec4f(rtc::halfToFloat(v[0]), rtc::halfToFloat(v[1]),
                     rtc::halfToFloat(v[2]), rtc::halfToFloat(v[3]));
      }
      default:
        return vec4f(0.f, 0.f, 0.f, 0.f);
      };
    }

  }
}
