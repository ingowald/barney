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
#include "barney/render/math.h"

namespace barney {
  namespace render {

    struct DG {
      vec3f Ng, Ns;
      vec3f wo;
      bool  insideMedium;
    };

    struct EvalRes {
      inline __device__ EvalRes() {}
      inline __device__ EvalRes(vec3f v, float p) : value(v),pdf(p) {}
      static inline __device__ EvalRes zero() { return { vec3f(0.f),0.f }; }
      vec3f value;
      float pdf;
    };
    
    
    struct SampleRes {
      // inline __device__ SampleRes() {}
      // inline __device__ SampleRes(vec3f v, float p) : value(v),pdf(p) {}
      static inline __device__ SampleRes zero() { return { vec3f(0.f), vec3f(0.f), 0, 0.f }; }
      vec3f weight;
      vec3f wi;
      int   type;
      float pdf;
    };
    
    
    inline __device__
    float luminance(vec3f c)
    { return 0.212671f*c.x + 0.715160f*c.y + 0.072169f*c.z; }



    inline __device__ 
    vec3f cartesian(float phi, float sinTheta, float cosTheta)
    {
      float sinPhi, cosPhi;
      sincosf(phi, &sinPhi, &cosPhi);
      return vec3f(cosPhi * sinTheta,
                   sinPhi * sinTheta,
                   cosTheta);
    }
    
    inline __device__ 
    vec3f cartesian(const float phi, const float cosTheta)
    {
      return cartesian(phi, cos2sin(cosTheta), cosTheta);
    }
    


    
    inline __device__ 
    vec3f cosineSampleHemisphere(const vec2f s)
    {
      const float phi = two_pi * s.x;
      const float cosTheta = sqrtf(s.y);
      const float sinTheta = sqrtf(1.0f - s.y);
      return cartesian(phi, sinTheta, cosTheta);
    }
    
    inline __device__ 
    float cosineSampleHemispherePDF(const vec3f &dir)
    {
      return dir.z * one_over_pi;
    }

    inline __device__ 
    float cosineSampleHemispherePDF(float cosTheta)
    {
      return cosTheta * one_over_pi;
    }
    
  }
}
