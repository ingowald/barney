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

// some functions tkaen from OSPRay, under this lincense:
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

namespace barney {
  namespace render {

#define one_over_pi (float(1.f/M_PI))
#define two_pi (float(2.f*M_PI))

    
    struct DG {
      vec3f N;
      vec3f wo;
    };

    struct EvalRes {
      inline __device__ EvalRes() {}
      inline __device__ EvalRes(vec3f v, float p) : value(v),pdf(p) {}
      vec3f value;
      float pdf;
    };
    
    

    inline __device__ float clamp(float f) { return min(1.f,max(0.f,f)); }
    inline __device__ float pow(float a, float b) { return powf(a,b); }
    inline __device__ float sqrt(float f) { return sqrtf(f); }
    inline __device__ float sqr(float f) { return f*f; }
    inline __device__ float cos2sin(const float f) { return sqrt(max(0.f, 1.f - sqr(f))); }
    inline __device__ float sin2cos(const float f) { return cos2sin(f); }
    



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
      const float cosTheta = sqrt(s.y);
      const float sinTheta = sqrt(1.0f - s.y);
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
