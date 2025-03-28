#pragma once

#include "barney/common/barney-common.h"
#ifdef __CUDACC__
#include <cuda/std/limits>
#endif
#include <limits>

#define ONE_PI ((float)M_PI)
#define TWO_PI (2.f*ONE_PI)
#define FOUR_PI (4.f*ONE_PI)
#define ONE_OVER_PI (1.f/ONE_PI)
#define ONE_OVER_TWO_PI (1.f/TWO_PI)
#define ONE_OVER_FOUR_PI (1.f/FOUR_PI)


#ifdef __CUDA_ARCH__
# define BARNEY_INF ::cuda::std::numeric_limits<float>::infinity()
#else
# define BARNEY_INF std::numeric_limits<float>::infinity()
#endif

namespace BARNEY_NS {
  inline __both__ float sqr(float f) { return f*f; }
  inline __both__ float cos2sin(const float f) { return sqrtf(max(0.f, 1.f - sqr(f))); }
  inline __both__ float sin2cos(const float f) { return cos2sin(f); }

  // ------------------------------------------------------------------
  // saturate - clamp to [0,1] range
  // ------------------------------------------------------------------
  inline __both__ float saturate(float f)
  { return max(0.f,min(f,1.f)); }
  
  inline __both__ vec3f saturate(vec3f v)
  { return vec3f{saturate(v.x),saturate(v.y),saturate(v.z)}; }

  inline __both__ vec4f saturate(vec4f v)
  { return vec4f{saturate(v.x),saturate(v.y),saturate(v.z),saturate(v.w)}; }

  // ------------------------------------------------------------------
  // linear_to_srgb conversion
  // ------------------------------------------------------------------

  inline __both__ float linear_to_srgb(float x)
  {
    if (x <= 0.0031308f) {
      return 12.92f * x;
    }
    return 1.055f * powf(x, 1.f/2.4f) - 0.055f;
  }

  inline __both__ vec3f linear_to_srgb(vec3f v)
  { return vec3f{linear_to_srgb(v.x),linear_to_srgb(v.y),linear_to_srgb(v.z)}; }

  /*! does linear-to-srgb conversion ON THE RGB CHANNELS of given
      vec4f. alpha remains unchanged */
  inline __both__ vec4f linear_to_srgb(vec4f v)
  { return vec4f{linear_to_srgb(v.x),linear_to_srgb(v.y),linear_to_srgb(v.z),v.w}; }

  // ------------------------------------------------------------------
  // lerp_l/lerp_r - linear interpolation
  // ------------------------------------------------------------------
  
  inline __both__ float lerp_r(float a, float b, float factor) { return (1.f-factor)*a+factor*b; }

  inline __both__ vec3f lerp_r(vec3f a, vec3f b, vec3f factor) { return (1.f-factor)*a+factor*b; }

  inline __both__ float lerp_l(float factor, float a, float b) { return (1.f-factor)*a+factor*b; }
  inline __both__ vec3f lerp_l(vec3f factor, vec3f a, vec3f b) { return (1.f-factor)*a+factor*b; }

  inline __both__ vec3f lerp_r(box3f box, vec3f f)
  { return lerp_l(f,box.lower,box.upper); }
  
  inline __both__ vec3f lerp_l(vec3f f, box3f box)
  { return lerp_l(f,box.lower,box.upper); }



  inline __both__ vec3f neg(vec3f v) { return vec3f(-v.x,-v.y,-v.z); }


  inline __both__
  float safeDiv(float a, float b) { return (b==0.f)?0.f:(a/b); }
  


  inline __both__ uint32_t make_8bit(const float f)
  {
    return min(255,max(0,int(f*256.f)));
  }

  inline __both__ uint32_t make_rgba(const vec3f color)
  {
    return
      (make_8bit(color.x) << 0) +
      (make_8bit(color.y) << 8) +
      (make_8bit(color.z) << 16) +
      (0xffU << 24);
  }
  inline __both__ uint32_t make_rgba(const vec4f color)
  {
    return
      (make_8bit(color.x) << 0) +
      (make_8bit(color.y) << 8) +
      (make_8bit(color.z) << 16) +
      (make_8bit(color.w) << 24);
  }

  // inline __both__ uint32_t make_rgba(const float4 color)
  // {
  //   return
  //     (make_8bit(color.x) << 0) +
  //     (make_8bit(color.y) << 8) +
  //     (make_8bit(color.z) << 16) +
  //     (make_8bit(color.w) << 24);
  // }

  inline __both__ float clamp(float f, float lo=0.f, float hi=1.f)
  { return min(hi,max(lo,f)); }
  inline __both__ int clamp(int f, int lo, int hi)
  {
      return min(hi, max(lo, f));
  }

}
