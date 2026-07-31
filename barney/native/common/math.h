// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/common/barney-common.h"
#if defined(__CUDACC__) && !defined(CUDART_INF)
#include <cuda/std/limits>
#endif
#include <limits>

#define ONE_PI ((float)M_PI)
#define TWO_PI (2.f*ONE_PI)
#define FOUR_PI (4.f*ONE_PI)
#define ONE_OVER_PI (1.f/ONE_PI)
#define ONE_OVER_TWO_PI (1.f/TWO_PI)
#define ONE_OVER_FOUR_PI (1.f/FOUR_PI)

#ifdef __HIPCC__
# define BARNEY_INF INFINITY
#elif defined(__CUDA_ARCH__)
// # define BARNEY_INF INFINITY
# define BARNEY_INF ::cuda::std::numeric_limits<float>::infinity()
#else
# define BARNEY_INF std::numeric_limits<float>::infinity()
#endif

namespace BARNEY_NS {
  namespace native {
    
    struct mat4f {
      float e[16];

      static mat4f identity();
    };

    using owl::common::min;
  
    inline __rtc_both float sqr(float f) { return f*f; }
    inline __rtc_both float cos2sin(const float f) { return sqrtf(max(0.f, 1.f - sqr(f))); }
    inline __rtc_both float sin2cos(const float f) { return cos2sin(f); }

    // ------------------------------------------------------------------
    // saturate - clamp to [0,1] range
    // ------------------------------------------------------------------
    inline __rtc_both float saturate(float f)
    { return max(0.f,min(f,1.f)); }
  
    inline __rtc_both vec3f saturate(vec3f v)
    { return vec3f{saturate(v.x),saturate(v.y),saturate(v.z)}; }

    inline __rtc_both vec4f saturate(vec4f v)
    { return vec4f{saturate(v.x),saturate(v.y),saturate(v.z),saturate(v.w)}; }

    // ------------------------------------------------------------------
    // linear_to_srgb conversion
    // ------------------------------------------------------------------

    inline __rtc_both float linear_to_srgb(float x)
    {
      if (x <= 0.0031308f) {
        return 12.92f * x;
      }
      return 1.055f * powf(x, 1.f/2.4f) - 0.055f;
    }

    inline __rtc_both vec3f linear_to_srgb(vec3f v)
    { return vec3f{linear_to_srgb(v.x),linear_to_srgb(v.y),linear_to_srgb(v.z)}; }

    /*! does linear-to-srgb conversion ON THE RGB CHANNELS of given
      vec4f. alpha remains unchanged */
    inline __rtc_both vec4f linear_to_srgb(vec4f v)
    { return vec4f{linear_to_srgb(v.x),linear_to_srgb(v.y),linear_to_srgb(v.z),v.w}; }

    // ------------------------------------------------------------------
    // lerp_l/lerp_r - linear interpolation
    // ------------------------------------------------------------------
  
    inline __rtc_both float lerp_r(float a, float b, float factor) { return (1.f-factor)*a+factor*b; }

    inline __rtc_both vec3f lerp_r(vec3f a, vec3f b, vec3f factor) { return (1.f-factor)*a+factor*b; }

    inline __rtc_both float lerp_l(float factor, float a, float b) { return (1.f-factor)*a+factor*b; }
    inline __rtc_both vec3f lerp_l(vec3f factor, vec3f a, vec3f b) { return (1.f-factor)*a+factor*b; }
    inline __rtc_both vec4f lerp_l(vec4f factor, vec4f a, vec4f b) { return (1.f-factor)*a+factor*b; }

    inline __rtc_both vec3f lerp_r(box3f box, vec3f f)
    { return lerp_l(f,box.lower,box.upper); }
  
    inline __rtc_both vec3f lerp_l(vec3f f, box3f box)
    { return lerp_l(f,box.lower,box.upper); }



    inline __rtc_both vec3f neg(vec3f v) { return vec3f(-v.x,-v.y,-v.z); }


    inline __rtc_both
    float safeDiv(float a, float b) { return (b==0.f)?0.f:(a/b); }
  


    inline __rtc_both uint32_t make_8bit(const float f)
    {
      return min(255,max(0,int(f*256.f)));
    }

    inline __rtc_both uint32_t make_rgba(const vec3f color)
    {
      return
        (make_8bit(color.x) << 0) +
        (make_8bit(color.y) << 8) +
        (make_8bit(color.z) << 16) +
        (0xffU << 24);
    }
    inline __rtc_both uint32_t make_rgba(const vec4f color)
    {
      return
        (make_8bit(color.x) << 0) +
        (make_8bit(color.y) << 8) +
        (make_8bit(color.z) << 16) +
        (make_8bit(color.w) << 24);
    }

    inline __rtc_both float clamp(float f, float lo=0.f, float hi=1.f)
    { return min(hi,max(lo,f)); }
    
    inline __rtc_both int clamp(int f, int lo, int hi)
    {
      return min(hi, max(lo, f));
    }

    inline __rtc_both vec4f make_vec4f(vec3f v, float w=1.f)
    { return {v.x,v.y,v.z,w}; }

    inline __rtc_both
    static mat4f identity()
    {
      mat4f res;
      
      res.e[ 0] = 1.f;
      res.e[ 1] = 0.f;
      res.e[ 2] = 0.f;
      res.e[ 3] = 0.f;

      res.e[ 4] = 0.f;
      res.e[ 5] = 1.f;
      res.e[ 6] = 0.f;
      res.e[ 7] = 0.f;

      res.e[ 8] = 0.f;
      res.e[ 9] = 0.f;
      res.e[10] = 1.f;
      res.e[11] = 0.f;

      res.e[12] = 0.f;
      res.e[13] = 0.f;
      res.e[14] = 0.f;
      res.e[15] = 1.f;
      return res;
    }
    
    inline __rtc_both
    vec4f operator*(const mat4f &m, const vec4f &v)
    {
      auto dot = [](vec4f a, vec4f b) {
        return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
      };
      return {
        dot(vec4f(m.e[ 0],m.e[ 4],m.e[ 8],m.e[12]),v),
        dot(vec4f(m.e[ 1],m.e[ 5],m.e[ 9],m.e[13]),v),
        dot(vec4f(m.e[ 2],m.e[ 6],m.e[10],m.e[14]),v),
        dot(vec4f(m.e[ 3],m.e[ 7],m.e[11],m.e[15]),v)
      };
    }

    inline std::ostream &operator<<(std::ostream &out, const mat4f &m)
    {
      out << '(' << m.e[ 0] << ',' << m.e[ 1] << ',' << m.e[ 2] << ',' << m.e[ 3] << ')'
          << '(' << m.e[ 4] << ',' << m.e[ 5] << ',' << m.e[ 6] << ',' << m.e[ 7] << ')'
          << '(' << m.e[ 8] << ',' << m.e[ 9] << ',' << m.e[10] << ',' << m.e[11] << ')'
          << '(' << m.e[12] << ',' << m.e[13] << ',' << m.e[14] << ',' << m.e[15] << ')';
      return out;
    }
    
  }
}
