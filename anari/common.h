// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sstream>
#include <vector>
#include "helium/array/Array.h"
#include "barney_math.h"
#include <iostream>
#include <cassert>

#ifndef PRINT
# define PRINT(var) std::cout << #var << "=" << var << std::endl;
#ifdef __WIN32__
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ << std::endl;
#else
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__ << std::endl;
#endif
#endif

namespace barney_device {

enum Attribute {
  Attribute0, Attribute1, Attribute2, Attribute3, Color, None=-1,
};

enum WrapMode {
  Clamp, Wrap, Mirror,
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

inline WrapMode toWrapMode(std::string str) {
  if (str == "clampToEdge")
    return Clamp;
  else if (str == "repeat")
    return Wrap;
  else if (str == "mirrorRepeat")
    return Mirror;

  return Clamp;
}

inline BNTextureAddressMode toBarneyAddressMode(std::string str) {
  if (str == "clampToEdge")
    return BN_TEXTURE_CLAMP;
  else if (str == "repeat")
    return BN_TEXTURE_WRAP;
  else if (str == "mirrorRepeat")
    return BN_TEXTURE_MIRROR;

  return BN_TEXTURE_CLAMP;
}

inline void convert_fixed8_to_float4(const void *in, math::float4 *out, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    out[i].x = ((const unsigned char *)in)[i] / 255.f;
    out[i].y = 0.f;
    out[i].z = 0.f;
    out[i].w = 1.f;
  }
}

inline void convert_fixed8x2_to_float4(const void *in, math::float4 *out, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    out[i].x = ((const math::byte2 *)in)[i].x / 255.f;
    out[i].y = ((const math::byte2 *)in)[i].y / 255.f;
    out[i].z = 0.f;
    out[i].w = 1.f;
  }
}

inline void convert_fixed8x3_to_float4(const void *in, math::float4 *out, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    out[i].x = ((const math::byte3 *)in)[i].x / 255.f;
    out[i].y = ((const math::byte3 *)in)[i].y / 255.f;
    out[i].z = ((const math::byte3 *)in)[i].z / 255.f;
    out[i].w = 1.f;
  }
}

inline void convert_fixed8x4_to_float4(const void *in, math::float4 *out, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    out[i].x = ((const math::byte4 *)in)[i].x / 255.f;
    out[i].y = ((const math::byte4 *)in)[i].y / 255.f;
    out[i].z = ((const math::byte4 *)in)[i].z / 255.f;
    out[i].w = ((const math::byte4 *)in)[i].w / 255.f;
  }
}

inline void convert_fixed16_to_float4(const void *in, math::float4 *out, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    out[i].x = ((const unsigned short *)in)[i] / 65535.f;
    out[i].y = 0.f;
    out[i].z = 0.f;
    out[i].w = 1.f;
  }
}

inline void convert_fixed16x2_to_float4(const void *in, math::float4 *out, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    out[i].x = ((const math::ushort2 *)in)[i].x / 65535.f;
    out[i].y = ((const math::ushort2 *)in)[i].y / 65535.f;
    out[i].z = 0.f;
    out[i].w = 1.f;
  }
}

inline void convert_fixed16x3_to_float4(const void *in, math::float4 *out, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    out[i].x = ((const math::ushort3 *)in)[i].x / 65535.f;
    out[i].y = ((const math::ushort3 *)in)[i].y / 65535.f;
    out[i].z = ((const math::ushort3 *)in)[i].z / 65535.f;
    out[i].w = 1.f;
  }
}

inline void convert_fixed16x4_to_float4(const void *in, math::float4 *out, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    out[i].x = ((const math::ushort4 *)in)[i].x / 65535.f;
    out[i].y = ((const math::ushort4 *)in)[i].y / 65535.f;
    out[i].z = ((const math::ushort4 *)in)[i].z / 65535.f;
    out[i].w = ((const math::ushort4 *)in)[i].w / 65535.f;
  }
}

inline void convert_float1_to_float4(const float *in, math::float4 *out, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    out[i].x = in[i];
    out[i].y = 0.f;
    out[i].z = 0.f;
    out[i].w = 1.f;
  }
}

inline void convert_float2_to_float4(const math::float2 *in, math::float4 *out, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    out[i].x = in[i].x;
    out[i].y = in[i].y;
    out[i].z = 0.f;
    out[i].w = 1.f;
  }
}

inline void convert_float3_to_float4(const math::float3 *in, math::float4 *out, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    out[i].x = in[i].x;
    out[i].y = in[i].y;
    out[i].z = in[i].z;
    out[i].w = 1.f;
  }
}

inline bool convert_to_float4(
    const helium::IntrusivePtr<helium::Array> &input, std::vector<math::float4> &data)
{
  data.resize(input->totalSize());
  if (input->elementType() == ANARI_UFIXED8) {
    convert_fixed8_to_float4(input->data(), data.data(), data.size());
  }
  else if (input->elementType() == ANARI_UFIXED8_VEC2) {
    convert_fixed8x2_to_float4(input->data(), data.data(), data.size());
  }
  else if (input->elementType() == ANARI_UFIXED8_VEC3) {
    convert_fixed8x3_to_float4(input->data(), data.data(), data.size());
  }
  else if (input->elementType() == ANARI_UFIXED8_VEC4) {
    convert_fixed8x4_to_float4(input->data(), data.data(), data.size());
  }
  else if (input->elementType() == ANARI_UFIXED8_RGBA_SRGB) {
    // this is WRONG in that it ignores the SRGB conversion
    convert_fixed8x4_to_float4(input->data(), data.data(), data.size());
  }
  else if (input->elementType() == ANARI_UFIXED8_RGB_SRGB) {
    // this is WRONG in that it ignores the SRGB conversion
    convert_fixed8x3_to_float4(input->data(), data.data(), data.size());
  }
  else if (input->elementType() == ANARI_UFIXED16) {
    convert_fixed16_to_float4(input->data(), data.data(), data.size());
  }
  else if (input->elementType() == ANARI_UFIXED16_VEC2) {
    convert_fixed16x2_to_float4(input->data(), data.data(), data.size());
  }
  else if (input->elementType() == ANARI_UFIXED16_VEC3) {
    convert_fixed16x3_to_float4(input->data(), data.data(), data.size());
  }
  else if (input->elementType() == ANARI_UFIXED16_VEC4) {
    convert_fixed16x4_to_float4(input->data(), data.data(), data.size());
  }
  else if (input->elementType() == ANARI_FLOAT32) {
    convert_float1_to_float4(input->dataAs<float>(), data.data(), data.size());
  }
  else if (input->elementType() == ANARI_FLOAT32_VEC2) {
    convert_float2_to_float4(input->dataAs<math::float2>(), data.data(), data.size());
  }
  else if (input->elementType() == ANARI_FLOAT32_VEC3) {
    convert_float3_to_float4(input->dataAs<math::float3>(), data.data(), data.size());
  }
  else if (input->elementType() == ANARI_FLOAT32_VEC4) {
    memcpy(data.data(), input->data(), input->totalSize() * sizeof(math::float4));
  }
  else {
    return false;
  }
  return true;
}

inline uint32_t make_8bit(const float f)
{
  return fminf(255,fmaxf(0,int(f*256.f)));
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

static BNData makeBarneyData(
    BNContext context, int slot, const helium::IntrusivePtr<helium::Array> &input)
{
  BNData res{0};

  std::vector<math::float4> data;

  if (input) {
    if (input->elementType() == ANARI_FLOAT32_VEC4) {
      res = bnDataCreate(context, slot, BN_FLOAT4, input->totalSize(), input->data());
    }
    else if (convert_to_float4(input, data) && !data.empty()) {
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
      res = bnTexture2DCreate(context, slot, BN_TEXEL_FORMAT_RGBA8,
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
