// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include "helium/array/Array1D.h"
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

static BNData makeBarneyData(
    BNModel model, int slot, const helium::IntrusivePtr<helium::Array1D> &input)
{
  BNData res{0};

  if (input) {
    if (input->elementType() == ANARI_UFIXED8_VEC4) {
      std::vector<math::float4> data(input->size(), math::float4(0.f, 0.f, 0.f, 1.f));
      for (size_t i = 0; i < data.size(); ++i) {
        data[i].x = ((const math::byte4 *)input->data())[i].x / 255.f;
        data[i].y = ((const math::byte4 *)input->data())[i].y / 255.f;
        data[i].z = ((const math::byte4 *)input->data())[i].z / 255.f;
        data[i].w = ((const math::byte4 *)input->data())[i].w / 255.f;
      }
      res = bnDataCreate(model, slot, BN_FLOAT4, data.size(), data.data());
    }
    else if (input->elementType() == ANARI_FLOAT32) {
      std::vector<math::float4> data(input->size(), math::float4(0.f, 0.f, 0.f, 1.f));
      for (size_t i = 0; i < data.size(); ++i) {
        data[i].x = input->beginAs<float>()[i];
      }
      res = bnDataCreate(model, slot, BN_FLOAT4, data.size(), data.data());
    }
    else if (input->elementType() == ANARI_FLOAT32_VEC2) {
      std::vector<math::float4> data(input->size(), math::float4(0.f, 0.f, 0.f, 1.f));
      for (size_t i = 0; i < data.size(); ++i) {
        data[i].x = input->beginAs<math::float2>()[i].x;
        data[i].y = input->beginAs<math::float2>()[i].y;
      }
      res = bnDataCreate(model, slot, BN_FLOAT4, data.size(), data.data());
    }
    else if (input->elementType() == ANARI_FLOAT32_VEC3) {
      std::vector<math::float4> data(input->size(), math::float4(0.f, 0.f, 0.f, 1.f));
      for (size_t i = 0; i < data.size(); ++i) {
        data[i].x = input->beginAs<math::float3>()[i].x;
        data[i].y = input->beginAs<math::float3>()[i].y;
        data[i].z = input->beginAs<math::float3>()[i].z;
      }
      res = bnDataCreate(model, slot, BN_FLOAT4, data.size(), data.data());
    }
    else if (input->elementType() == ANARI_FLOAT32_VEC4) {
      res = bnDataCreate(model, slot, BN_FLOAT4, input->size(), input->begin());
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
