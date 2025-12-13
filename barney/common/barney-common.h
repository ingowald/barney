// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/barney.h"
#include "barney/barneyConfig.h"

#if BARNEY_HAVE_HIP
# define CUDART_INF INFINITY
# define CUDART_INF_F ((float)INFINITY)
# define CUDART_NAN NAN
# define CUDART_NAN_F ((float)NAN)
# include "hip/hip_runtime.h"
#endif

// automatically generated, in build dir
#include "rtcore/AppInterface.h"
#include "barney/api/common.h"
#include <owl/common/owl-common.h>
#include <owl/common/math/box.h>
#include <owl/common/math/AffineSpace.h>
#include <owl/common/math/random.h>
// #ifdef __CUDACC__
# define OWL_DISABLE_TBB
// #endif
#include <owl/common/parallel/parallel_for.h>
// #include "rtcore/ComputeInterface.h"

#define __barney_align(a) OWL_ALIGN(a)

#if BARNEY_MPI
# if BARNEY_RTC_OPTIX
#  define BARNEY_MPI_NS barney_optix
# endif
# if BARNEY_RTC_EMBREE
#  define BARNEY_MPI_NS barney_embree
# endif
# if BARNEY_RTC_CUDA
#  define BARNEY_MPI_NS barney_cuda
# endif
#else
# if BARNEY_RTC_OPTIX
#  define BARNEY_NS barney_optix
# endif
# if BARNEY_RTC_EMBREE
#  define BARNEY_NS barney_embree
# endif
# if BARNEY_RTC_CUDA
#  define BARNEY_NS barney_cuda
# endif
#endif

namespace BARNEY_NS {
  
  using namespace owl::common;
  typedef owl::common::interval<float> range1f;
  // using Random = LCG<8>;

  /*! divRoundUp, but always in 64-bits, to avoid type conflicts */
  inline __rtc_both size_t dru(size_t a, size_t b)
  { return (a+b-1)/b; }
                                      
  struct RNGSeed {
    inline __rtc_both void seed(uint32_t a, uint32_t b)
    {
      state ^= 2*a+1;
      next();
      state ^= 2*b;
      next();
    }
      
    inline __rtc_both uint32_t next()
    {
      uint64_t x = state;
      uint64_t const multiplier = 6364136223846793005ull;
      unsigned count = (unsigned)(x >> 61);
      state = x * multiplier;
      x ^= x >> 22;
      return (uint32_t)(x >> (22 + count));	        
    }
    inline __rtc_both uint32_t next(uint32_t hash)
    {
      return next((uint64_t)hash);
    }
    inline __rtc_both uint32_t next(uint64_t hash)
    {
      state ^= (hash<<1);
      // next((uint32_t)hash);
      // next((uint32_t)(hash>>32));
      return next();
    }
    uint64_t state = 0xcafef00dd15ea5e5u;
  };
  
  struct Random2 : public RNGSeed {
    inline __rtc_both Random2(uint32_t a,
                              uint32_t b)
    { seed(a,b); }
    inline __rtc_both Random2(RNGSeed &seed,
                              uint32_t b)
    { this->state = seed.state; seed.next(); this->next(b ? b : 290374); }
    inline __rtc_both Random2(RNGSeed &seed,
                              uint64_t b)
    { this->state = seed.state; seed.next(); this->next(b ? b : 290374); }
    inline __rtc_both float operator()()
    {
      return (next() & 0x00FFFFFF) * (1.f / (float) 0x01000000);
    };
  };

  
  template<typename T>
  inline __both__
  void swap(T &a, T &b) { T c = a; a = b; b = c; }

  /*! helper function to extrace 3f spatial component from 4f point-plus-scalar */
  inline __both__ vec3f getPos(vec4f v)
  {return vec3f{v.x,v.y,v.z}; }

  /*! helper function to extrace 3f spatial component from 4f point-plus-scalar */
  inline __both__ box3f getBox(box4f bb)
  { return box3f{getPos(bb.lower),getPos(bb.upper)}; }
  inline __both__ box3f getBox(box3f bb)
  { return bb; }

  /*! helper function to extract 1f scalar range from 4f point-plus-scalar */
  inline __both__ range1f getRange(box4f bb)
  { return range1f{bb.lower.w,bb.upper.w}; }

  inline __both__ uint64_t hash(uint32_t v)
  {
    const uint64_t FNV_offset_basis = 0xcbf29ce484222325ULL;
    const uint64_t FNV_prime = 0x100000001b3ULL;
    return FNV_offset_basis ^ FNV_prime * v;
  }
  inline __both__ uint64_t hash(uint64_t h, uint32_t v)
  {
    const uint64_t FNV_prime = 0x100000001b3ULL;
    return h * FNV_prime ^ v;
  }
  
  inline __both__ uint64_t hash(uint32_t v0, uint32_t v1)
  { return hash(hash(v0),v1); }
  
  inline __both__ uint64_t hash(uint32_t v0, uint32_t v1, uint32_t v2)
  { return hash(hash(v0,v1),v2); }
  inline __both__ uint64_t hash(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3)
  { return hash(hash(v0,v1,v2),v3); }
  inline __both__ uint64_t hash(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, uint32_t v4)
  { return hash(hash(v0,v1,v2,v3),v4); }

  // using rtc::optix::printDev;
}

#define BARNEY_NYI() throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" not yet implemented")

#define BARNEY_INVALID_VALUE() throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" invalid or un-implemented switch value")

