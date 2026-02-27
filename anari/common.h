// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney_math.h"
// anari
#include "anari/frontend/type_utility.h"
// helium
#include "helium/array/Array.h"
// std
#include <iostream>
#include <cassert>
#include <limits>
#include <sstream>
#include <vector>

#define BANARI_TRACK_LEAKS(a) /* nothing */

namespace barney_device {

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

template <int TYPE>
inline void convert_to_float4(const void *_in, std::vector<math::float4> &out)
{
  using T = typename anari::ANARITypeProperties<TYPE>::base_type;
  const auto components = anari::ANARITypeProperties<TYPE>::components;
  const T *in = (const T *)_in;
  for (size_t i = 0; i < out.size(); ++i)
    anari::ANARITypeProperties<TYPE>::toFloat4(&out[i].x, &in[i * components]);
}

inline bool convert_to_float4(
    const helium::IntrusivePtr<helium::Array> &input, std::vector<math::float4> &data)
{
  data.resize(input->totalSize());
  switch(input->elementType()) {
    case ANARI_FIXED8:
      convert_to_float4<ANARI_FIXED8>(input->data(), data);
      return true;
    case ANARI_FIXED8_VEC2:
      convert_to_float4<ANARI_FIXED8_VEC2>(input->data(), data);
      return true;
    case ANARI_FIXED8_VEC3:
      convert_to_float4<ANARI_FIXED8_VEC3>(input->data(), data);
      return true;
    case ANARI_FIXED8_VEC4:
      convert_to_float4<ANARI_FIXED8_VEC4>(input->data(), data);
      return true;
    case ANARI_FIXED16:
      convert_to_float4<ANARI_FIXED16>(input->data(), data);
      return true;
    case ANARI_FIXED16_VEC2:
      convert_to_float4<ANARI_FIXED16_VEC2>(input->data(), data);
      return true;
    case ANARI_FIXED16_VEC3:
      convert_to_float4<ANARI_FIXED16_VEC3>(input->data(), data);
      return true;
    case ANARI_FIXED16_VEC4:
      convert_to_float4<ANARI_FIXED16_VEC4>(input->data(), data);
      return true;
    case ANARI_FIXED32:
      convert_to_float4<ANARI_FIXED32>(input->data(), data);
      return true;
    case ANARI_FIXED32_VEC2:
      convert_to_float4<ANARI_FIXED32_VEC2>(input->data(), data);
      return true;
    case ANARI_FIXED32_VEC3:
      convert_to_float4<ANARI_FIXED32_VEC3>(input->data(), data);
      return true;
    case ANARI_FIXED32_VEC4:
      convert_to_float4<ANARI_FIXED32_VEC4>(input->data(), data);
      return true;
    case ANARI_UFIXED8_R_SRGB:
      convert_to_float4<ANARI_UFIXED8_R_SRGB>(input->data(), data);
      return true;
    case ANARI_UFIXED8_RA_SRGB:
      convert_to_float4<ANARI_UFIXED8_RA_SRGB>(input->data(), data);
      return true;
    case ANARI_UFIXED8_RGB_SRGB:
      convert_to_float4<ANARI_UFIXED8_RGB_SRGB>(input->data(), data);
      return true;
    case ANARI_UFIXED8_RGBA_SRGB:
      convert_to_float4<ANARI_UFIXED8_RGBA_SRGB>(input->data(), data);
      return true;
    case ANARI_UFIXED8:
      convert_to_float4<ANARI_UFIXED8>(input->data(), data);
      return true;
    case ANARI_UFIXED8_VEC2:
      convert_to_float4<ANARI_UFIXED8_VEC2>(input->data(), data);
      return true;
    case ANARI_UFIXED8_VEC3:
      convert_to_float4<ANARI_UFIXED8_VEC3>(input->data(), data);
      return true;
    case ANARI_UFIXED8_VEC4:
      convert_to_float4<ANARI_UFIXED8_VEC4>(input->data(), data);
      return true;
    case ANARI_UFIXED16:
      convert_to_float4<ANARI_UFIXED16>(input->data(), data);
      return true;
    case ANARI_UFIXED16_VEC2:
      convert_to_float4<ANARI_UFIXED16_VEC2>(input->data(), data);
      return true;
    case ANARI_UFIXED16_VEC3:
      convert_to_float4<ANARI_UFIXED16_VEC3>(input->data(), data);
      return true;
    case ANARI_UFIXED16_VEC4:
      convert_to_float4<ANARI_UFIXED16_VEC4>(input->data(), data);
      return true;
    case ANARI_UFIXED32:
      convert_to_float4<ANARI_UFIXED32>(input->data(), data);
      return true;
    case ANARI_UFIXED32_VEC2:
      convert_to_float4<ANARI_UFIXED32_VEC2>(input->data(), data);
      return true;
    case ANARI_UFIXED32_VEC3:
      convert_to_float4<ANARI_UFIXED32_VEC3>(input->data(), data);
      return true;
    case ANARI_UFIXED32_VEC4:
      convert_to_float4<ANARI_UFIXED32_VEC4>(input->data(), data);
      return true;
    case ANARI_FLOAT32:
      convert_to_float4<ANARI_FLOAT32>(input->data(), data);
      return true;
    case ANARI_FLOAT32_VEC2:
      convert_to_float4<ANARI_FLOAT32_VEC2>(input->data(), data);
      return true;
    case ANARI_FLOAT32_VEC3:
      convert_to_float4<ANARI_FLOAT32_VEC3>(input->data(), data);
      return true;
    case ANARI_FLOAT32_VEC4:
      convert_to_float4<ANARI_FLOAT32_VEC4>(input->data(), data);
      return true;
    case ANARI_FLOAT64:
      convert_to_float4<ANARI_FLOAT64>(input->data(), data);
      return true;
    case ANARI_FLOAT64_VEC2:
      convert_to_float4<ANARI_FLOAT64_VEC2>(input->data(), data);
      return true;
    case ANARI_FLOAT64_VEC3:
      convert_to_float4<ANARI_FLOAT64_VEC3>(input->data(), data);
      return true;
    case ANARI_FLOAT64_VEC4:
      convert_to_float4<ANARI_FLOAT64_VEC4>(input->data(), data);
      return true;
    default:
      break;
  }

  return false;
}

inline uint32_t make_8bit(const float f)
{
  return (uint32_t)fminf(255.f,fmaxf(0.f,f*256.f));
}

inline uint32_t make_rgba8(const math::float4 color)
{
  return
    (make_8bit(color.x) << 0) +
    (make_8bit(color.y) << 8) +
    (make_8bit(color.z) << 16) +
    (make_8bit(color.w) << 24);
}

inline bool convert_to_rgba8(
    const helium::IntrusivePtr<helium::Array> &input, std::vector<uint32_t> &data)
{
  std::vector<math::float4> rgba32f;
  if (convert_to_float4(input, rgba32f)) {
    data.resize(rgba32f.size());
    for (size_t i = 0; i < rgba32f.size(); ++i) {
      data[i] = make_rgba8(rgba32f[i]);
    }
    return true;
  }
  return false;
}

static BNData makeBarneyData(BNContext context,
                             int slot,
                             const helium::IntrusivePtr<helium::Array> &input,
                             helium::BaseObject *warnObject
                             )
{
  BNData res{0};

  std::vector<math::float4> data;

  if (input && input->totalSize() > 0) {
    if (input->elementType() == ANARI_FLOAT32_VEC4) {
      res = bnDataCreate(context, slot, BN_FLOAT4, input->totalSize(), input->data());
    }
    else if (convert_to_float4(input, data) && !data.empty()) {
      warnObject->reportMessage(ANARI_SEVERITY_DEBUG,
                                "makeBarneyData converts %s to float4",
                                anari::toString(input->elementType()));
      res = bnDataCreate(context, slot, BN_FLOAT4, data.size(), data.data());
    }
    else {
      std::stringstream ss;
      ss << "unsupported element type: "
         << anari::toString(input->elementType());
      std::string str = ss.str();
      fprintf(stderr, "%s\n", str.c_str());

    }
  }

  return res;
}

static BNTexture2D makeBarneyTexture2D(
    BNContext context, int slot, const helium::IntrusivePtr<helium::Array> &input,
    int width, int height,
    BNTextureFilterMode filterMode = BN_TEXTURE_LINEAR,
    BNTextureAddressMode addressMode = BN_TEXTURE_CLAMP)
{
  BNTexture2D res{0};

  std::vector<uint32_t> texels;

  if (input) {
    if (convert_to_rgba8(input, texels)) {
      res = bnTexture2DCreate(context, slot, BN_UFIXED8_RGBA,
                              width, height, texels.data(),
                              filterMode, addressMode);
    }
    else {
      std::stringstream ss;
      ss << "unsupported texel type: "
         << anari::toString(input->elementType());
      std::string str = ss.str();
      fprintf(stderr, "%s\n", str.c_str());
    }
  }

  return res;
}

} // namespace barney_device
