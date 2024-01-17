// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

// barney
#include "barney.h"
// anari
#include <helium/helium_math.h>
#include <anari/anari_cpp.hpp>
// std
#include <cmath>
#include <cstring> // for std::memcpy()

namespace barney_device {

namespace math = anari::math;

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
  math::float3 lower, upper;

  box3() { invalidate(); }
  box3(const math::float3 &l, const math::float3 &u) : lower(l), upper(u) {}

  void invalidate()
  {
    lower = math::float3(INFINITY, INFINITY, INFINITY);
    upper = math::float3(-INFINITY, -INFINITY, -INFINITY);
  }

  box3 &insert(math::float3 v)
  {
    lower.x = fminf(lower.x, v.x);
    lower.y = fminf(lower.y, v.y);
    lower.z = fminf(lower.z, v.z);
    upper.x = fmaxf(upper.x, v.x);
    upper.y = fmaxf(upper.y, v.y);
    upper.z = fmaxf(upper.z, v.z);
    return *this;
  }

  box3 &insert(const box3 b)
  {
    insert(b.lower);
    insert(b.upper);
    return *this;
  }
};

struct box3i
{
  math::int3 lower, upper;
};

} // namespace barney_device

///////////////////////////////////////////////////////////////////////////////
// ANARITypeFor type trait mappings ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace anari {

ANARI_TYPEFOR_SPECIALIZATION(barney_device::box1, ANARI_FLOAT32_BOX1);
ANARI_TYPEFOR_SPECIALIZATION(barney_device::box3, ANARI_FLOAT32_BOX3);
ANARI_TYPEFOR_SPECIALIZATION(barney_device::box3i, ANARI_INT32_BOX3);

#ifdef ANARI_BARNEY_MATH_DEFINITIONS
ANARI_TYPEFOR_DEFINITION(barney_device::box1);
ANARI_TYPEFOR_DEFINITION(barney_device::box3);
ANARI_TYPEFOR_DEFINITION(barney_device::box3i);
#endif

} // namespace anari
