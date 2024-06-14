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

#include "SheenDistribution.h"

namespace barney {
  namespace render {

    // GGX (Trowbridge-Reitz) microfacet distribution
    // [Walter et al., 2007, "Microfacet Models for Refraction through Rough
    // Surfaces"] [Heitz, 2014, "Understanding the Masking-Shadowing Function in
    // Microfacet-Based BRDFs"] [Heitz and d'Eon, 2014, "Importance Sampling
    // Microfacet-Based BSDFs using the Distribution of Visible Normals"] [Heitz,
    // 2017, "A Simpler and Exact Sampling Routine for the GGX Distribution of
    // Visible Normals"]
    struct GGXDistribution
    {
      inline static __device__ GGXDistribution create(vec2f alpha);
      

      inline __device__ vec3f sampleVisible(const vec3f &wo, float &pdf, const vec2f &s) const;
      inline __device__ float evalVisible(const vec3f &wh,
                                          const vec3f &wo,
                                          float cosThetaOH,
                                          float &pdf) const;
      inline __device__ float evalG2(const vec3f &wo,
                                     const vec3f &wi,
                                     float cosThetaOH,
                                     float cosThetaIH) const;
      inline __device__ float evalLambda(const vec3f &wo) const;
      inline __device__ float evalG1(const vec3f &wo, float cosThetaOH) const;
      inline __device__ vec3f sample(float &pdf, const vec2f &s) const;
      inline __device__ float eval(const vec3f &wh) const;
      inline __device__ float eval(const vec3f &wh, float &pdf) const;

      vec2f alpha;
    };

    inline __device__ GGXDistribution GGXDistribution::create(vec2f alpha)
    {
      GGXDistribution self;
      self.alpha = alpha;
      return self;
    }

    // D(\omega_m) = \frac{1}{\pi \alpha_x \alpha_y \cos^4\theta_m \left(1 +
    // \tan^2\theta_m \left(\frac{\cos^2\phi_m}{\alpha_x^2} +
    // \frac{\sin^2\phi_m}{\alpha_y^2}\right)\right)^2}
    inline __device__ float GGXDistribution::eval(const vec3f &wh) const
    {
      float cosTheta = wh.z;
      float cosTheta2 = sqr(cosTheta);

      float e = (sqr(wh.x / alpha.x) + sqr(wh.y / alpha.y)) / cosTheta2;
      return rcp(
                 (float)pi * alpha.x * alpha.y * sqr(cosTheta2 * (1.f + e)));
    }

    // p(\omega_m) = D(\omega_m) \cos\theta_m
    inline __device__ float GGXDistribution::eval(const vec3f &wh, float &pdf) const
    {
      float cosTheta = wh.z;
      float D = eval(wh);
      pdf = D * fabsf(cosTheta);
      return D;
    }
    
    // \theta_m = \arctan \left( \frac{\alpha\sqrt{\xi_1}}{\sqrt{1-\xi_1}} \right)
    // \phi_m   = 2\pi \xi_2
    // p(\omega_m) = D(\omega_m) \cos\theta_m
    inline __device__ vec3f GGXDistribution::sample(float &pdf, const vec2f &s) const
    {
      float phi;
      if (alpha.x == alpha.y) {
        phi = 2.f * (float)pi * s.y;
      } else {
        phi =
          atan(alpha.y / alpha.x * tan((float)pi * (2.f * s.y + 0.5f)));
        if (s.y > 0.5f)
          phi += (float)pi;
      }

      float sinPhi, cosPhi;
      sincos(phi, &sinPhi, &cosPhi);

      float alpha2;
      if (alpha.x == alpha.y)
        alpha2 = sqr(alpha.x);
      else
        alpha2 = rcp(sqr(cosPhi / alpha.x) + sqr(sinPhi / alpha.y));

      float tanTheta2 = alpha2 * s.x / (1.f - s.x);
      float cosTheta = rsqrt(1.f + tanTheta2);
      float cosTheta3 = sqr(cosTheta) * cosTheta;
      float sinTheta = cos2sin(cosTheta);

      float e = tanTheta2 / alpha2;
      pdf = rcp((float)pi * alpha.x * alpha.y * cosTheta3 * sqr(1.f + e));

      float x = cosPhi * sinTheta;
      float y = sinPhi * sinTheta;
      float z = cosTheta;
      return vec3f(x, y, z);
    }

    // Smith Lambda function [Heitz, 2014]
    // \Lambda(\omega_o) = \frac{-1 + \sqrt{1+\frac{1}{a^2}}}{2}
    // a = \frac{1}{\alpha_o \tan\theta_o}
    // \alpha_o = \sqrt{cos^2\phi_o \alpha_x^2 + sin^2\phi_o \alpha_y^2}
    inline __device__ float GGXDistribution::evalLambda(const vec3f &wo) const
    {
      float cosThetaO = wo.z;
      float cosThetaO2 = sqr(cosThetaO);
      float invA2 =
        (sqr(wo.x * alpha.x) + sqr(wo.y * alpha.y)) / cosThetaO2;
      return 0.5f * (-1.f + sqrt(1.f + invA2));
    }

    inline __device__ float GGXDistribution::evalG1(const vec3f &wo, float cosThetaOH) const
    {
      float cosThetaO = wo.z;
      if (cosThetaO * cosThetaOH <= 0.f)
        return 0.f;

      return rcp(1.f + evalLambda(wo));
    }

    // Smith's height-correlated masking-shadowing function
    // [Heitz, 2014, "Understanding the Masking-Shadowing Function in
    // Microfacet-Based BRDFs"]
    inline __device__ float GGXDistribution::evalG2(const vec3f &wo,
                                         const vec3f &wi,
                                         float cosThetaOH,
                                         float cosThetaIH) const
    {
      float cosThetaO = wo.z;
      float cosThetaI = wi.z;
      if (cosThetaO * cosThetaOH <= 0.f || cosThetaI * cosThetaIH <= 0.f)
        return 0.f;

      return rcp(1.f + evalLambda(wo) + evalLambda(wi));
    }

    inline __device__ float GGXDistribution::evalVisible(const vec3f &wh,
                                              const vec3f &wo,
                                              float cosThetaOH,
                                              float &pdf) const
    {
      float cosThetaO = wo.z;
      float D = eval(wh);
      pdf = evalG1(wo, cosThetaOH) * fabsf(cosThetaOH) * D / fabsf(cosThetaO);
      return D;
    }

    // Fast visible normal sampling (wo must be in the upper hemisphere)
    // [Heitz, 2017, "A Simpler and Exact Sampling Routine for the GGX Distribution
    // of Visible Normals"]
    inline __device__ vec3f GGXDistribution::sampleVisible(const vec3f &wo, float &pdf, const vec2f &s) const
    {
      // Stretch wo
      vec3f V =
        normalize(vec3f(alpha.x * wo.x, alpha.y * wo.y, wo.z));

      // Orthonormal basis
      vec3f T1 = (V.z < 0.9999f) ? normalize(cross(V, vec3f(0, 0, 1)))
        : vec3f(1.f, 0.f, 0.f);
      vec3f T2 = cross(T1, V);

      // Sample point with polar coordinates (r, phi)
      float a = 1.f / (1.f + V.z);
      float r = sqrt(s.x);
      float phi = (s.y < a) ? s.y / a * (float)pi
        : (float)pi + (s.y - a) / (1.f - a) * (float)pi;
      float P1 = r * cos(phi);
      float P2 = r * sin(phi) * ((s.y < a) ? 1.f : V.z);

      // Compute normal
      vec3f wh = P1 * T1 + P2 * T2 + sqrt(max(0.f, 1.f - P1 * P1 - P2 * P2)) * V;

      // Unstretch
      wh = normalize(vec3f(alpha.x * wh.x, alpha.y * wh.y, max(0.f, wh.z)));

      // Compute pdf
      float cosThetaO = wo.z;
      pdf = evalG1(wo, dot(wo, wh)) * fabsf(dot(wo, wh)) * eval(wh)
        / fabsf(cosThetaO);
      return wh;
    }

  }
}
