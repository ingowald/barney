// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

// barney
#include "barney/barney.h"
#if BARNEY_MPI
# include "barney/barney_mpi.h"
#endif
// anari
#include <helium/helium_math.h>
#include <anari/anari_cpp.hpp>
// std
#include <cmath>
#include <cstring> // for std::memcpy()
#include <iostream>
#ifdef __CUDACC__
#include <cuda/std/limits>
#endif

#ifdef __CUDACC__
# define BARNEY_INF ::cuda::std::numeric_limits<float>::infinity()
#else
# define BARNEY_INF INFINITY
#endif
  
namespace barney_device {

  namespace math = anari::math;

  inline void bnSet3ic(BNObject o, const char *n, math::int3 v)
  { bnSet3i(o,n,v.x,v.y,v.z); }
  inline void bnSet3fc(BNObject o, const char *n, math::float3 v)
  { bnSet3f(o,n,v.x,v.y,v.z); }
  inline void bnSet4fc(BNObject o, const char *n, math::float4 v)
  { bnSet4f(o,n,v.x,v.y,v.z,v.z); }

  
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
      lower = math::float3(BARNEY_INF, BARNEY_INF, BARNEY_INF);
      upper = math::float3(-BARNEY_INF, -BARNEY_INF, -BARNEY_INF);
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

  inline std::ostream &operator<<(std::ostream &o, bn_float3 v)
  {
    o << "(" << v.x << "," << v.y << "," << v.z << ")";
    return o;
  }
  inline std::ostream &operator<<(std::ostream &o, bn_float4 v)
  {
    o << "(" << v.x << "," << v.y << "," << v.z << "," << v.w << ")";
    return o;
  }
  inline std::ostream &operator<<(std::ostream &o, anari::math::float3 v)
  {
    o << "(" << v.x << "," << v.y << "," << v.z << ")";
    return o;
  }
  inline std::ostream &operator<<(std::ostream &o, anari::math::float4 v)
  {
    o << "(" << v.x << "," << v.y << "," << v.z << "," << v.w << ")";
    return o;
  }
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

#ifndef PRINT
#define PRINT(var) std::cout << #var << "=" << var << std::endl;
#ifdef __WIN32__
#define PING                                                            \
  std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__     \
  << std::endl;
#else
#define PING                                                            \
  std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__ \
  << std::endl;
#endif
#endif

  
} // namespace anari
