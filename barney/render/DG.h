// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


// some functions taken from OSPRay, under this lincense:
// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "barney/common/barney-common.h"
#include "barney/common/half.h"
#include "barney/common/Texture.h"
#include "barney/common/Data.h"
#include "barney/common/math.h"

namespace BARNEY_NS {
  namespace render {
    using Random = BARNEY_NS::Random2;
    
    struct DG {
      vec3f Ng, Ns;
      vec3f wo;
      vec3f P;
      bool  insideMedium;
    };

    struct EvalRes {
      inline __both__ EvalRes() {}
      inline __rtc_device EvalRes(vec3f v, float p) : value(v),pdf(p) {}
      static inline __rtc_device EvalRes zero() { return { vec3f(0.f),0.f }; }
      inline __rtc_device bool valid() const    { return pdf > 0.f; };// && !isinf(pdf) }
      vec3f value;
      float pdf;
    };

    /* TODO merge with ScatterResult - this is leftovers from ospray glass import */
    struct SampleRes {
      typedef enum { INVALID, DIFFUSE_REFLECTION, SPECULAR_REFLECTION, SPECULAR_TRANSMISSION } Type;
      
      static inline __rtc_device SampleRes zero()
      { return { vec3f(0.f), vec3f(0.f), INVALID, 0.f }; }
      inline __rtc_device bool valid() const    { return pdf > 0.f; };
      vec3f weight;
      vec3f wi;
      Type  type;
      float pdf;
    };

    /*! result of scattering a ray on a differential surface,
        according to a BSDF */
    struct ScatterResult {
      typedef enum {
        INVALID  = 0,
        NONE,
        DIFFUSE,
        SPECULAR,
        GLOSSY,
        VOLUME,
      } Type;
      inline __rtc_device bool valid() const    { return pdf > 0.f; }

      vec3f f_r;
      vec3f dir;
      float pdf  = 0.f;
      bool  changedMedium = false;
      float offsetDirection = +1.f;
      ScatterResult::Type type = ScatterResult::INVALID;
      // bool  wasDiffuse = true;;
    };


    inline __rtc_device
    vec3f sampleCosineWeightedHemisphere(vec3f Ns, Random &random)
    {
      while (1) {
        vec3f p = 2.f*vec3f(random(),random(),random()) - vec3f(1.f);
        if (dot(p,p) > 1.f) continue;
        return normalize(normalize(p)+Ns);
      }
    }

    inline __rtc_device float pbrt_clampf(float f, float lo, float hi)
    { return max(lo,min(hi,f)); }

    inline __rtc_device float pbrtSphericalTheta(const vec3f &v)
    {
      return acosf(pbrt_clampf(v.z, -1.f, 1.f));
    }

    inline __rtc_device float pbrtSphericalPhi(const vec3f &v)
    {
      float p = atan2f(v.y, v.x);
      return (p < 0.f) ? (p + float(2.f * M_PI)) : p;
    }

    inline __rtc_device
    float luminance(vec3f c)
    {
      return 0.212671f*c.x + 0.715160f*c.y + 0.072169f*c.z;
    }
    
    inline __rtc_device
    vec3f reflect(vec3f v, vec3f n)
    {
      return v - (2.f*dot(v,n))*n;
    }
    
    inline __rtc_device
    vec3f refract(vec3f v, vec3f n, float eta)
    {
      float dotValue = dot(n,v);
      float k = 1.f-eta*eta*(1.f-dotValue*dotValue);
      return (k >= 0.f)
        ? (eta*v - (eta*dotValue + sqrtf(k)*n))
        : vec3f(0.f);
    }

    inline __rtc_device
    bool all_zero(vec3f v) { return v.x==0 && v.y==0 && v.z == 0; }
    


    inline __rtc_device
    vec3f cartesian(float phi, float sinTheta, float cosTheta)
    {
      float sinPhi, cosPhi;
#ifdef _WIN32
      sinPhi = sinf(phi);
      cosPhi = cosf(phi);
#elif __APPLE__
      __sincosf(phi, &sinPhi, &cosPhi);
#else
      sincosf(phi, &sinPhi, &cosPhi);
#endif
      return vec3f(cosPhi * sinTheta,
                   sinPhi * sinTheta,
                   cosTheta);
    }

    inline __rtc_device
    vec3f cartesian(const float phi, const float cosTheta)
    {
      return cartesian(phi, cos2sin(cosTheta), cosTheta);
    }




    inline __rtc_device
    vec3f cosineSampleHemisphere(const vec2f s)
    {
      const float phi = TWO_PI * s.x;
      const float cosTheta = sqrtf(s.y);
      const float sinTheta = sqrtf(1.0f - s.y);
      return cartesian(phi, sinTheta, cosTheta);
    }

    inline __rtc_device
    float cosineSampleHemispherePDF(const vec3f &dir)
    {
      return dir.z * ONE_OVER_PI;
    }

    inline __rtc_device
    float cosineSampleHemispherePDF(float cosTheta)
    {
      return cosTheta * ONE_OVER_PI;
    }

    inline __rtc_device
    vec3f randomDirection(Random &rng)
    {
      vec3f v;
      while (true) {
        v.x = 1.f-2.f*rng();
        v.y = 1.f-2.f*rng();
        v.z = 1.f-2.f*rng();
        if (dot(v,v) <= 1.f)
          return normalize(v);
      }
    }

    
  }
}
