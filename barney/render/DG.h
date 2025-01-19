// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

#include "barney/common/Texture.h"
#include "barney/common/Data.h"
#include "barney/common/half.h"
#include "barney/render/floatN.h"

namespace barney {
  namespace render {

    struct DG {
      vec3f Ng, Ns;
      vec3f wo;
      vec3f P;
      bool  insideMedium;
    };

    struct EvalRes {
      inline __both__ EvalRes() {}
      inline __both__ EvalRes(vec3f v, float p) : value(v),pdf(p) {}
      static inline __both__ EvalRes zero() { return { vec3f(0.f),0.f }; }
      inline __both__ bool valid() const    { return pdf > 0.f; };// && !isinf(pdf) }
      vec3f value;
      float pdf;
    };


    struct SampleRes {
      // inline __both__ SampleRes() {}
      // inline __both__ SampleRes(vec3f v, float p) : value(v),pdf(p) {}
      static inline __both__ SampleRes zero() { return { vec3f(0.f), vec3f(0.f), 0, 0.f }; }
      inline __both__ bool valid() const    { return pdf > 0.f; };// && !isinf(pdf); }
      vec3f weight;
      vec3f wi;
      int   type;
      float pdf;
    };

    /*! result of scattering a ray on a differential surface,
        according to a BSDF */
    struct ScatterResult {
      // typedef enum {
      //   NONE=0,
      //   DIFFUSE  = (1<<0),
      //   SPECULAR = (1<<1),
      //   GLOSSY   = (1<<2),
      //   DIFFUSE_TRANS  = (1<<3),
      //   SPECULAR_TRANS = (1<<4),
      //   GLOSSY_TRANS   = (1<<5),
      // } Type;
      inline __both__ bool valid() const    { return pdf > 0.f
          // && !isinf(pdf)
          ;
      }

      vec3f f_r;
      vec3f dir;
      float pdf  = 0.f;
      // Type  type = NONE;
      bool  changedMedium = false;
      float offsetDirection = +1.f;
      bool  wasDiffuse;
    };


    inline __both__
    vec3f sampleCosineWeightedHemisphere(vec3f Ns, Random &random)
    {
      while (1) {
        vec3f p = 2.f*vec3f(random(),random(),random()) - vec3f(1.f);
        if (dot(p,p) > 1.f) continue;
        return normalize(normalize(p)
                         +Ns//vec3f(0.f,0.f,1.f)
                         );
      }
    }

  inline __both__ float pbrt_clampf(float f, float lo, float hi)
  { return max(lo,min(hi,f)); }

  inline __both__ float pbrtSphericalTheta(const vec3f &v)
  {
    return acosf(pbrt_clampf(v.z, -1.f, 1.f));
  }

  inline __both__ float pbrtSphericalPhi(const vec3f &v)
  {
    float p = atan2f(v.y, v.x);
    return (p < 0.f) ? (p + float(2.f * M_PI)) : p;
  }


    inline __both__
    float luminance(vec3f c)
    { return 0.212671f*c.x + 0.715160f*c.y + 0.072169f*c.z; }



    inline __both__
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

    inline __both__
    vec3f cartesian(const float phi, const float cosTheta)
    {
      return cartesian(phi, cos2sin(cosTheta), cosTheta);
    }




    inline __both__
    vec3f cosineSampleHemisphere(const vec2f s)
    {
      const float phi = two_pi * s.x;
      const float cosTheta = sqrtf(s.y);
      const float sinTheta = sqrtf(1.0f - s.y);
      return cartesian(phi, sinTheta, cosTheta);
    }

    inline __both__
    float cosineSampleHemispherePDF(const vec3f &dir)
    {
      return dir.z * one_over_pi;
    }

    inline __both__
    float cosineSampleHemispherePDF(float cosTheta)
    {
      return cosTheta * one_over_pi;
    }

  }
}
