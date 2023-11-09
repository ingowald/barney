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
