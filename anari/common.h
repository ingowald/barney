// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include "helium/array/Array.h"
#include "barney_math.h"

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

inline void convert_fixed4_to_float4(const void *in, math::float4 *out, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    out[i].x = ((const math::byte4 *)in)[i].x / 255.f;
    out[i].y = ((const math::byte4 *)in)[i].y / 255.f;
    out[i].z = ((const math::byte4 *)in)[i].z / 255.f;
    out[i].w = ((const math::byte4 *)in)[i].w / 255.f;
  }
}

inline void convert_float1_to_float4(const float *in, math::float4 *out, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    out[i].x = in[i];
    out[i].y = 0.f;
    out[i].z = 0.f;
    out[i].w = 0.f;
  }
}

inline void convert_float2_to_float4(const math::float2 *in, math::float4 *out, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    out[i].x = in[i].x;
    out[i].y = in[i].y;
    out[i].z = 0.f;
    out[i].w = 0.f;
  }
}

inline void convert_float3_to_float4(const math::float3 *in, math::float4 *out, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    out[i].x = in[i].x;
    out[i].y = in[i].y;
    out[i].z = in[i].z;
    out[i].w = 0.f;
  }
}

static BNData makeBarneyData(
    BNModel model, int slot, const helium::IntrusivePtr<helium::Array> &input)
{
  BNData res{0};

  if (input) {
    if (input->elementType() == ANARI_UFIXED8_VEC4) {
      std::vector<math::float4> data(input->totalSize());
      convert_fixed4_to_float4(input->data(), data.data(), data.size());
      res = bnDataCreate(model, slot, BN_FLOAT4, data.size(), data.data());
    }
    else if (input->elementType() == ANARI_FLOAT32) {
      std::vector<math::float4> data(input->totalSize());
      convert_float1_to_float4(input->dataAs<float>(), data.data(), data.size());
      res = bnDataCreate(model, slot, BN_FLOAT4, data.size(), data.data());
    }
    else if (input->elementType() == ANARI_FLOAT32_VEC2) {
      std::vector<math::float4> data(input->totalSize());
      convert_float2_to_float4(input->dataAs<math::float2>(), data.data(), data.size());
      res = bnDataCreate(model, slot, BN_FLOAT4, data.size(), data.data());
    }
    else if (input->elementType() == ANARI_FLOAT32_VEC3) {
      std::vector<math::float4> data(input->totalSize());
      convert_float3_to_float4(input->dataAs<math::float3>(), data.data(), data.size());
      res = bnDataCreate(model, slot, BN_FLOAT4, data.size(), data.data());
    }
    else if (input->elementType() == ANARI_FLOAT32_VEC4) {
      res = bnDataCreate(model, slot, BN_FLOAT4, input->totalSize(), input->data());
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

} // namespace barney_device
