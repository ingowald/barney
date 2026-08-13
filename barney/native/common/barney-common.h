// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney_rtc.h"
#include "rtcore/common/rtcore-common.h"
#include "native/include/barney.h"
 
namespace BARNEY_NS {
  namespace native {
    
    using namespace owl::common;
    typedef owl::common::interval<float> range1f;

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
    inline __rtc_both
    void swap(T &a, T &b) { T c = a; a = b; b = c; }

    /*! helper function to extrace 3f spatial component from 4f point-plus-scalar */
    inline __rtc_both vec3f getPos(vec4f v)
    {return vec3f{v.x,v.y,v.z}; }

    /*! helper function to extrace 3f spatial component from 4f point-plus-scalar */
    inline __rtc_both box3f getBox(box4f bb)
    { return box3f{getPos(bb.lower),getPos(bb.upper)}; }
    inline __rtc_both box3f getBox(box3f bb)
    { return bb; }

    /*! helper function to extract 1f scalar range from 4f point-plus-scalar */
    inline __rtc_both range1f getRange(box4f bb)
    { return range1f{bb.lower.w,bb.upper.w}; }

    inline __rtc_both uint64_t hash(uint32_t v)
    {
      const uint64_t FNV_offset_basis = 0xcbf29ce484222325ULL;
      const uint64_t FNV_prime = 0x100000001b3ULL;
      return FNV_offset_basis ^ FNV_prime * v;
    }
    inline __rtc_both uint64_t hash(uint64_t h, uint32_t v)
    {
      const uint64_t FNV_prime = 0x100000001b3ULL;
      return h * FNV_prime ^ v;
    }
  
    inline __rtc_both uint64_t hash(uint32_t v0, uint32_t v1)
    { return hash(hash(v0),v1); }
  
    inline __rtc_both uint64_t hash(uint32_t v0, uint32_t v1, uint32_t v2)
    { return hash(hash(v0,v1),v2); }
    inline __rtc_both uint64_t hash(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3)
    { return hash(hash(v0,v1,v2),v3); }
    inline __rtc_both uint64_t hash(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, uint32_t v4)
    { return hash(hash(v0,v1,v2,v3),v4); }

  }
}

#define BARNEY_NYI() throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" not yet implemented")

#define BARNEY_INVALID_VALUE() throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" invalid or un-implemented switch value")

