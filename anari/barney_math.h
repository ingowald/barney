// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

// barney
#include "barney.h"
// cuda
#include <vector_functions.hpp>
// anari
#include <anari/anari_cpp.hpp>
// std
#include <cmath>
#include <cstring> // for std::memcpy()

namespace anari {

struct box1
{
  float lower, upper;
  box1 &insert(float v)
  {
    lower = fminf(lower, v);
    upper = fmaxf(upper, v);
    return *this;
  }
};

struct box3
{
  float3 lower, upper;
  box3 &insert(float3 v)
  {
    lower.x = fminf(lower.x, v.x);
    lower.y = fminf(lower.y, v.y);
    lower.z = fminf(lower.z, v.z);
    upper.x = fmaxf(lower.x, v.x);
    upper.y = fmaxf(lower.y, v.y);
    upper.z = fmaxf(lower.z, v.z);
    return *this;
  }
};

struct box3i
{
  int3 lower, upper;
};

ANARI_TYPEFOR_SPECIALIZATION(uchar2, ANARI_UINT8_VEC2);
ANARI_TYPEFOR_SPECIALIZATION(uchar3, ANARI_UINT8_VEC3);
ANARI_TYPEFOR_SPECIALIZATION(uchar4, ANARI_UINT8_VEC4);
ANARI_TYPEFOR_SPECIALIZATION(int2, ANARI_INT32_VEC2);
ANARI_TYPEFOR_SPECIALIZATION(int3, ANARI_INT32_VEC3);
ANARI_TYPEFOR_SPECIALIZATION(int4, ANARI_INT32_VEC4);
ANARI_TYPEFOR_SPECIALIZATION(uint2, ANARI_UINT32_VEC2);
ANARI_TYPEFOR_SPECIALIZATION(uint3, ANARI_UINT32_VEC3);
ANARI_TYPEFOR_SPECIALIZATION(uint4, ANARI_UINT32_VEC4);
ANARI_TYPEFOR_SPECIALIZATION(float2, ANARI_FLOAT32_VEC2);
ANARI_TYPEFOR_SPECIALIZATION(float3, ANARI_FLOAT32_VEC3);
ANARI_TYPEFOR_SPECIALIZATION(float4, ANARI_FLOAT32_VEC4);
ANARI_TYPEFOR_SPECIALIZATION(box1, ANARI_FLOAT32_BOX1);
ANARI_TYPEFOR_SPECIALIZATION(box3, ANARI_FLOAT32_BOX3);
ANARI_TYPEFOR_SPECIALIZATION(box3i, ANARI_INT32_BOX3);

#ifdef ANARI_BARNEY_MATH_DEFINITIONS
ANARI_TYPEFOR_DEFINITION(uchar2);
ANARI_TYPEFOR_DEFINITION(uchar3);
ANARI_TYPEFOR_DEFINITION(uchar4);
ANARI_TYPEFOR_DEFINITION(int2);
ANARI_TYPEFOR_DEFINITION(int3);
ANARI_TYPEFOR_DEFINITION(int4);
ANARI_TYPEFOR_DEFINITION(uint2);
ANARI_TYPEFOR_DEFINITION(uint3);
ANARI_TYPEFOR_DEFINITION(uint4);
ANARI_TYPEFOR_DEFINITION(float2);
ANARI_TYPEFOR_DEFINITION(float3);
ANARI_TYPEFOR_DEFINITION(float4);
ANARI_TYPEFOR_DEFINITION(box1);
ANARI_TYPEFOR_DEFINITION(box3);
ANARI_TYPEFOR_DEFINITION(box3i);
#endif

} // namespace anari

namespace barney_device {

inline float dot(float3 a, float3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float length2(float3 a)
{
  return dot(a, a);
}

inline float length(float3 a)
{
  return std::sqrt(length2(a));
}

inline float3 normalize(float3 a)
{
  auto l = length(a);
  return make_float3(a.x / l, a.y / l, a.z / l);
}

} // namespace barney_device
