// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney_math.h"
// anari
#include "anari/frontend/type_utility.h"
// helium
#include "helium/array/Array.h"
// std
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#define BANARI_TRACK_LEAKS(a) /* nothing */

namespace BARNEY_NS {
  namespace anari {

    enum Attribute {
      Attribute0, Attribute1, Attribute2, Attribute3, Color, None=-1,
    };

    inline Attribute toAttribute(std::string str) {
      if (str == "attribute0")
        return Attribute0;
      else if (str == "attribute1")
        return Attribute1;
      else if (str == "attribute2")
        return Attribute2;
      else if (str == "attribute3")
        return Attribute3;
      else if (str == "color")
        return Color;
      else if (str == "none")
        return None;
      return None;
    }

    inline BNTextureAddressMode toBarneyAddressMode(std::string str) {
      if (str == "clampToEdge")
        return BN_TEXTURE_CLAMP;
      else if (str == "repeat")
        return BN_TEXTURE_WRAP;
      else if (str == "mirrorRepeat")
        return BN_TEXTURE_MIRROR;
      else if (str == "clampToBorder")
        return BN_TEXTURE_BORDER;

      return BN_TEXTURE_CLAMP;
    }

    // ------------------------------------------------------------------
    // upload-time demotion: the only sanctioned host conversion.
    // Formats no backend can carry are narrowed to float32 of the same
    // vec width. this is exact for sources with at most 24 significant
    // bits (all 8/16-bit types) and LOSSY beyond that - float64 and
    // the 32/64-bit fixed types lose precision here. float16 is NOT
    // demoted - it is native everywhere; the half decode lives in
    // rtcore (rtc::halfToFloat) and the device typedRead.
    // ------------------------------------------------------------------

    template <int TYPE>
    inline void demote_componentwise(const void *_in,
                                     std::vector<float> &data)
    {
      using Props = ::anari::ANARITypeProperties<TYPE>;
      using T = typename Props::base_type;
      const int components = Props::components;
      const T *in = (const T *)_in;
      const size_t N = data.size() / components;
      for (size_t i = 0; i < N; ++i) {
        float tmp[4];
        Props::toFloat4(tmp, &in[i * components]);
        for (int c = 0; c < components; ++c)
          data[i * components + c] = tmp[c];
      }
    }

    template <int TYPE>
    inline void demote_componentwise4(const void *_in,
                                      std::vector<math::float4> &data)
    {
      using Props = ::anari::ANARITypeProperties<TYPE>;
      using T = typename Props::base_type;
      const int components = Props::components;
      const T *in = (const T *)_in;
      for (size_t i = 0; i < data.size(); ++i) {
        float tmp[4];
        Props::toFloat4(tmp, &in[i * components]);
        data[i] = math::float4(tmp[0], tmp[1], tmp[2], tmp[3]);
      }
    }

    /*! demote to float32 of the same vec width. returns the component
      count, or 0 if the type is not demotable */
    inline int demoteToF32N(const helium::IntrusivePtr<helium::Array> &input,
                            std::vector<float> &data)
    {
      const auto type = input->elementType();
      const int components = ::anari::componentsOf(type);
      const size_t N = input->totalSize();
      if (N == 0 || components == 0)
        return 0;

#define DEMOTE_CASE(TYPE)                       \
      case TYPE:                                 \
        data.resize(N * components);            \
        demote_componentwise<TYPE>(input->data(), data); \
        return components;

      switch (type) {
        DEMOTE_CASE(ANARI_FIXED8)
        DEMOTE_CASE(ANARI_FIXED8_VEC2)
        DEMOTE_CASE(ANARI_FIXED8_VEC3)
        DEMOTE_CASE(ANARI_FIXED8_VEC4)
        DEMOTE_CASE(ANARI_FIXED16)
        DEMOTE_CASE(ANARI_FIXED16_VEC2)
        DEMOTE_CASE(ANARI_FIXED16_VEC3)
        DEMOTE_CASE(ANARI_FIXED16_VEC4)
        DEMOTE_CASE(ANARI_FIXED32)
        DEMOTE_CASE(ANARI_FIXED32_VEC2)
        DEMOTE_CASE(ANARI_FIXED32_VEC3)
        DEMOTE_CASE(ANARI_FIXED32_VEC4)
        DEMOTE_CASE(ANARI_UFIXED32)
        DEMOTE_CASE(ANARI_UFIXED32_VEC2)
        DEMOTE_CASE(ANARI_UFIXED32_VEC3)
        DEMOTE_CASE(ANARI_UFIXED32_VEC4)
        DEMOTE_CASE(ANARI_UFIXED8_VEC3)
        DEMOTE_CASE(ANARI_UFIXED16_VEC3)
        DEMOTE_CASE(ANARI_UFIXED16_VEC4)
        DEMOTE_CASE(ANARI_UFIXED8_R_SRGB)
        // RA's toFloat4 is (srgb(r),0,0,linear(a)); keep all four
        // lanes so typedRead's vec2 fill cannot put alpha in G.
        case ANARI_UFIXED8_RA_SRGB: {
          data.resize(N * 4);
          const uint8_t *in = (const uint8_t *)input->data();
          for (size_t i = 0; i < N; ++i)
            ::anari::ANARITypeProperties<ANARI_UFIXED8_RA_SRGB>
              ::toFloat4(&data[i * 4], &in[i * 2]);
          return 4;
        }
        DEMOTE_CASE(ANARI_UFIXED8_RGB_SRGB)
        DEMOTE_CASE(ANARI_UFIXED8_RGBA_SRGB)
        DEMOTE_CASE(ANARI_FLOAT64)
        DEMOTE_CASE(ANARI_FLOAT64_VEC2)
        DEMOTE_CASE(ANARI_FLOAT64_VEC3)
        DEMOTE_CASE(ANARI_FLOAT64_VEC4)
        // float32/float16 sources are normally carried pristine, but
        // they reach here when rtcore has no texel format of that
        // width (eg 2-component float textures)
        DEMOTE_CASE(ANARI_FLOAT32)
        DEMOTE_CASE(ANARI_FLOAT32_VEC2)
        DEMOTE_CASE(ANARI_FLOAT32_VEC3)
        DEMOTE_CASE(ANARI_FLOAT32_VEC4)
        DEMOTE_CASE(ANARI_FLOAT16)
        DEMOTE_CASE(ANARI_FLOAT16_VEC2)
        DEMOTE_CASE(ANARI_FLOAT16_VEC3)
        DEMOTE_CASE(ANARI_FLOAT16_VEC4)
        DEMOTE_CASE(ANARI_INT8)
        DEMOTE_CASE(ANARI_INT8_VEC2)
        DEMOTE_CASE(ANARI_INT8_VEC3)
        DEMOTE_CASE(ANARI_INT8_VEC4)
        DEMOTE_CASE(ANARI_UINT8)
        DEMOTE_CASE(ANARI_UINT8_VEC2)
        DEMOTE_CASE(ANARI_UINT8_VEC3)
        DEMOTE_CASE(ANARI_UINT8_VEC4)
        DEMOTE_CASE(ANARI_INT16)
        DEMOTE_CASE(ANARI_INT16_VEC2)
        DEMOTE_CASE(ANARI_INT16_VEC3)
        DEMOTE_CASE(ANARI_INT16_VEC4)
        DEMOTE_CASE(ANARI_UINT16)
        DEMOTE_CASE(ANARI_UINT16_VEC2)
        DEMOTE_CASE(ANARI_UINT16_VEC3)
        DEMOTE_CASE(ANARI_UINT16_VEC4)
        DEMOTE_CASE(ANARI_INT32)
        DEMOTE_CASE(ANARI_INT32_VEC2)
        DEMOTE_CASE(ANARI_INT32_VEC3)
        DEMOTE_CASE(ANARI_INT32_VEC4)
        DEMOTE_CASE(ANARI_UINT32)
        DEMOTE_CASE(ANARI_UINT32_VEC2)
        DEMOTE_CASE(ANARI_UINT32_VEC3)
        DEMOTE_CASE(ANARI_UINT32_VEC4)
        DEMOTE_CASE(ANARI_INT64)
        DEMOTE_CASE(ANARI_INT64_VEC2)
        DEMOTE_CASE(ANARI_INT64_VEC3)
        DEMOTE_CASE(ANARI_INT64_VEC4)
        DEMOTE_CASE(ANARI_UINT64)
        DEMOTE_CASE(ANARI_UINT64_VEC2)
        DEMOTE_CASE(ANARI_UINT64_VEC3)
        DEMOTE_CASE(ANARI_UINT64_VEC4)
        DEMOTE_CASE(ANARI_FIXED64)
        DEMOTE_CASE(ANARI_FIXED64_VEC2)
        DEMOTE_CASE(ANARI_FIXED64_VEC3)
        DEMOTE_CASE(ANARI_FIXED64_VEC4)
        DEMOTE_CASE(ANARI_UFIXED64)
        DEMOTE_CASE(ANARI_UFIXED64_VEC2)
        DEMOTE_CASE(ANARI_UFIXED64_VEC3)
        DEMOTE_CASE(ANARI_UFIXED64_VEC4)
      default:
        break;
      }
#undef DEMOTE_CASE
      return 0;
    }

    /*! texture-flavored demotion: multi-component data is repacked to
      vec4 with ANARI fill semantics (the hardware fetches 1/2/4
      components only, so width-N targets are pointless here) */
    inline bool demoteToF32Vec4(const helium::IntrusivePtr<helium::Array> &input,
                                std::vector<math::float4> &data)
    {
      const auto type = input->elementType();
      const size_t N = input->totalSize();
      if (N == 0)
        return false;

#define DEMOTE4_CASE(TYPE)                     \
      case TYPE:                                 \
        data.resize(N);                         \
        demote_componentwise4<TYPE>(input->data(), data); \
        return true;

      switch (type) {
        DEMOTE4_CASE(ANARI_FIXED8)
        DEMOTE4_CASE(ANARI_FIXED8_VEC2)
        DEMOTE4_CASE(ANARI_FIXED8_VEC3)
        DEMOTE4_CASE(ANARI_FIXED8_VEC4)
        DEMOTE4_CASE(ANARI_FIXED16)
        DEMOTE4_CASE(ANARI_FIXED16_VEC2)
        DEMOTE4_CASE(ANARI_FIXED16_VEC3)
        DEMOTE4_CASE(ANARI_FIXED16_VEC4)
        DEMOTE4_CASE(ANARI_FIXED32)
        DEMOTE4_CASE(ANARI_FIXED32_VEC2)
        DEMOTE4_CASE(ANARI_FIXED32_VEC3)
        DEMOTE4_CASE(ANARI_FIXED32_VEC4)
        DEMOTE4_CASE(ANARI_UFIXED32)
        DEMOTE4_CASE(ANARI_UFIXED32_VEC2)
        DEMOTE4_CASE(ANARI_UFIXED32_VEC3)
        DEMOTE4_CASE(ANARI_UFIXED32_VEC4)
        DEMOTE4_CASE(ANARI_UFIXED8_VEC3)
        DEMOTE4_CASE(ANARI_UFIXED16_VEC3)
        DEMOTE4_CASE(ANARI_UFIXED16_VEC4)
        DEMOTE4_CASE(ANARI_INT8)
        DEMOTE4_CASE(ANARI_INT8_VEC2)
        DEMOTE4_CASE(ANARI_INT8_VEC3)
        DEMOTE4_CASE(ANARI_INT8_VEC4)
        DEMOTE4_CASE(ANARI_UINT8)
        DEMOTE4_CASE(ANARI_UINT8_VEC2)
        DEMOTE4_CASE(ANARI_UINT8_VEC3)
        DEMOTE4_CASE(ANARI_UINT8_VEC4)
        DEMOTE4_CASE(ANARI_INT16)
        DEMOTE4_CASE(ANARI_INT16_VEC2)
        DEMOTE4_CASE(ANARI_INT16_VEC3)
        DEMOTE4_CASE(ANARI_INT16_VEC4)
        DEMOTE4_CASE(ANARI_UINT16)
        DEMOTE4_CASE(ANARI_UINT16_VEC2)
        DEMOTE4_CASE(ANARI_UINT16_VEC3)
        DEMOTE4_CASE(ANARI_UINT16_VEC4)
        DEMOTE4_CASE(ANARI_INT32)
        DEMOTE4_CASE(ANARI_INT32_VEC2)
        DEMOTE4_CASE(ANARI_INT32_VEC3)
        DEMOTE4_CASE(ANARI_INT32_VEC4)
        DEMOTE4_CASE(ANARI_UINT32)
        DEMOTE4_CASE(ANARI_UINT32_VEC2)
        DEMOTE4_CASE(ANARI_UINT32_VEC3)
        DEMOTE4_CASE(ANARI_UINT32_VEC4)
        DEMOTE4_CASE(ANARI_INT64)
        DEMOTE4_CASE(ANARI_INT64_VEC2)
        DEMOTE4_CASE(ANARI_INT64_VEC3)
        DEMOTE4_CASE(ANARI_INT64_VEC4)
        DEMOTE4_CASE(ANARI_UINT64)
        DEMOTE4_CASE(ANARI_UINT64_VEC2)
        DEMOTE4_CASE(ANARI_UINT64_VEC3)
        DEMOTE4_CASE(ANARI_UINT64_VEC4)
        DEMOTE4_CASE(ANARI_FIXED64)
        DEMOTE4_CASE(ANARI_FIXED64_VEC2)
        DEMOTE4_CASE(ANARI_FIXED64_VEC3)
        DEMOTE4_CASE(ANARI_FIXED64_VEC4)
        DEMOTE4_CASE(ANARI_UFIXED64)
        DEMOTE4_CASE(ANARI_UFIXED64_VEC2)
        DEMOTE4_CASE(ANARI_UFIXED64_VEC3)
        DEMOTE4_CASE(ANARI_UFIXED64_VEC4)
        DEMOTE4_CASE(ANARI_FLOAT64)
        DEMOTE4_CASE(ANARI_FLOAT64_VEC2)
        DEMOTE4_CASE(ANARI_FLOAT64_VEC3)
        DEMOTE4_CASE(ANARI_FLOAT64_VEC4)
        // as above: float sources land here only when no native texel
        // format of that width exists
        DEMOTE4_CASE(ANARI_FLOAT32)
        DEMOTE4_CASE(ANARI_FLOAT32_VEC2)
        DEMOTE4_CASE(ANARI_FLOAT32_VEC3)
        DEMOTE4_CASE(ANARI_FLOAT32_VEC4)
        DEMOTE4_CASE(ANARI_FLOAT16)
        DEMOTE4_CASE(ANARI_FLOAT16_VEC2)
        DEMOTE4_CASE(ANARI_FLOAT16_VEC3)
        DEMOTE4_CASE(ANARI_FLOAT16_VEC4)
      default:
        break;
      }
#undef DEMOTE4_CASE
      return false;
    }

    // ------------------------------------------------------------------
    // host pad: widening that costs no precision, only an extra lane
    // ------------------------------------------------------------------

    /*! widen 3-component texels to 4 in their own precision, so a pad
      never costs precision or memory beyond the extra lane. the added
      lane is alpha=1 in the source encoding (ANARI fill is
      (0,0,0,1)); for normalized integer types that is the type's max
      code, for half it is 0x3C00 */
    inline void padToFourComponents(const helium::IntrusivePtr<helium::Array> &input,
                                    std::vector<uint8_t> &out,
                                    size_t &bytesPerTexel)
    {
      const auto type = input->elementType();
      const size_t N = input->totalSize();
      const size_t srcComponents = ::anari::componentsOf(type);
      const size_t scalarSize = ::anari::sizeOf(type) / srcComponents;
      bytesPerTexel = scalarSize * 4;

      out.resize(N * bytesPerTexel);
      const uint8_t *in = (const uint8_t *)input->data();
      for (size_t i = 0; i < N; ++i) {
        uint8_t *dst = out.data() + i * bytesPerTexel;
        memcpy(dst, in + i * scalarSize * srcComponents,
               scalarSize * srcComponents);
        uint8_t *alpha = dst + scalarSize * 3;
        switch (type) {
        case ANARI_FLOAT16_VEC3: {
          const uint16_t one = 0x3C00;
          memcpy(alpha, &one, 2);
        } break;
        case ANARI_FLOAT32_VEC3: {
          const float one = 1.f;
          memcpy(alpha, &one, 4);
        } break;
        case ANARI_UFIXED8_VEC3:
          *alpha = 255;
          break;
        case ANARI_UFIXED16_VEC3: {
          const uint16_t one = 65535;
          memcpy(alpha, &one, 2);
        } break;
        case ANARI_FIXED8_VEC3:
          *(int8_t *)alpha = 127;
          break;
        case ANARI_FIXED16_VEC3: {
          const int16_t one = 32767;
          memcpy(alpha, &one, 2);
        } break;
        default:
          memset(alpha, 0, scalarSize);
          break;
        }
      }
    }

    // same pad for a raw float3 buffer (e.g. the HDRI env map)
    inline void padFloat3ToVec4(const void *_in, size_t N,
                                std::vector<math::float4> &data)
    {
      const math::float3 *in = (const math::float3 *)_in;
      data.resize(N);
      for (size_t i = 0; i < N; ++i) {
        const math::float3 v = in[i];
        data[i] = math::float4(v.x, v.y, v.z, 1.f);
      }
    }

  }
}
