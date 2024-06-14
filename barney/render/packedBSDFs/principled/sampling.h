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

/*! taken partly from ospray, under following license:

// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

*/

#pragma once

#include "sampling.h"

namespace barney {
  namespace render {

      inline __device__ float one_over_two_pi() 
      { return float(1.f/(2.f*M_PI)); }
      
      // inline __device__ vec3f cartesian(const float phi, const float sinTheta, const float cosTheta)
      // {
      //   float sinPhi, cosPhi;
      //   sincos(phi, &sinPhi, &cosPhi);
      //   return vec3f(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
      // }
      
      // inline __device__ vec3f cartesian(const float phi, const float cosTheta)
      // {
      //   return cartesian(phi, cos2sin(cosTheta), cosTheta);
      // }

      
      inline __device__ vec3f uniformSampleHemisphere(const vec2f s)
      {
        const float phi = (float)two_pi * s.x;
        const float cosTheta = s.y;
        const float sinTheta = cos2sin(s.y);
        return cartesian(phi, sinTheta, cosTheta);
      }
      
      inline __device__ float uniformSampleHemispherePDF()
      {
        return one_over_two_pi();
      }
      
    // helper function which computes cosT^2 from cosI and eta
    inline __device__ float sqrCosT(const float cosI, const float eta)
    {
      return 1.0f - sqr(eta) * (1.0f - sqr(cosI));
    }
    
  }
}
