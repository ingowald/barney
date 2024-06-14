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

    inline __device__ float roughnessToAlpha(float roughness)
    {
      // Roughness is squared for perceptual reasons
      return max(sqr(roughness), 0.001f);
    }

    // [Burley, 2012, "Physically Based Shading at Disney", Course Notes, v3]
    inline __device__ vec2f roughnessToAlpha(float roughness, float anisotropy)
    {
      float aspect = sqrtf(1.f - 0.9f * anisotropy);
      return vec2f(max(sqr(roughness) / aspect, 0.001f),
                   max(sqr(roughness) * aspect, 0.001f));
    }

    
    // Cylindrical microfiber distribution for sheen
    // [Conty and Kulla, 2017, "Production Friendly Microfacet Sheen BRDF"]
    // [Kulla and Conty, 2017, "Revisiting Physically Based Shading at Imageworks"]
    struct SheenDistribution
    {
      static inline __device__
      SheenDistribution create(float r);


      inline __device__ float eval(float cosTheta) const;
      
      // Helper function for computing Lambda
      inline __device__ float evalL(float x) const;
      
      inline __device__ float evalLambda(float cosTheta) const;
      
      // Non-physical artistic adjustment for a softer light terminator
      inline __device__ float evalLambdaI(float cosTheta) const;
      
      // Smith's height-correlated masking-shadowing function
      // [Heitz, 2014, "Understanding the Masking-Shadowing Function in
      // Microfacet-Based BRDFs"]
      inline __device__ float evalG2(float cosThetaO,
                                     float cosThetaI,
                                     float cosThetaOH,
                                     float cosThetaIH) const;
      
      float r; // in (0, 1]
    };

    inline __device__ SheenDistribution SheenDistribution::create(float r)
    {
      SheenDistribution self;
      self.r = r;
      return self;
    }

    inline __device__ float SheenDistribution::eval(float cosTheta) const
    {
      float sinTheta = cos2sin(cosTheta);
      float invr = rcp(this->r);
      return (2.f + invr) * powf(sinTheta, invr) * one_over_two_pi();
    }

    // Helper function for computing Lambda
    inline __device__ float SheenDistribution::evalL(float x) const
    {
      const float a0 = 25.3245f;
      const float a1 = 21.5473f;
      const float b0 =  3.32435f;
      const float b1 =  3.82987f;
      const float c0 =  0.16801f;
      const float c1 =  0.19823f;
      const float d0 = -1.27393f;
      const float d1 = -1.97760f;
      const float e0 = -4.85967f;
      const float e1 = -4.32054f;

      float r  = this->r;
      float w0 = sqr(1.f - r);
      float w1 = 1.f - w0;

      float a = w0 * a0 + w1 * a1;
      float b = w0 * b0 + w1 * b1;
      float c = w0 * c0 + w1 * c1;
      float d = w0 * d0 + w1 * d1;
      float e = w0 * e0 + w1 * e1;

      return a / (1.f + b * pow(x, c)) + d * x + e;
    }

    inline __device__ float SheenDistribution::evalLambda(float cosTheta) const
    {
      if (cosTheta < 0.5f)
        return expf(evalL(cosTheta));
      else
        return expf(2.f * evalL(0.5f) - evalL(1.f - cosTheta));
    }

    // Non-physical artistic adjustment for a softer light terminator
    inline __device__ float SheenDistribution::evalLambdaI(float cosTheta) const
    {
      float x = 1.f + 2.f * sqr(sqr(sqr(1.f - cosTheta)));
      return pow(evalLambda(cosTheta), x);
    }

    // Smith's height-correlated masking-shadowing function
    // [Heitz, 2014, "Understanding the Masking-Shadowing Function in
    // Microfacet-Based BRDFs"]
    inline __device__ float SheenDistribution::evalG2(float cosThetaO,
                                                      float cosThetaI,
                                                      float cosThetaOH,
                                                      float cosThetaIH) const
    {
      if (cosThetaO * cosThetaOH <= 0.f || cosThetaI * cosThetaIH <= 0.f)
        return 0.f;

      // return rcp(1.f + evalLambda(self, cosThetaO) + evalLambda(self,
      // cosThetaI));
      return rcp(1.f + evalLambda(cosThetaO) + evalLambdaI(cosThetaI));
    }

  }
}
