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

#include "barney/material/device/DG.h"
#include "barney/material/device/BSDF.h"

namespace barney {
  namespace render {

    // inline __both__ float roughnessToAlpha(float roughness)
    // {
    //   // Roughness is squared for perceptual reasons
    //   // return max(sqr(roughness), 0.001f);
    //   return vec2f(max(sqr(roughness) / aspect, 0.001f),
    //                     max(sqr(roughness) * aspect, 0.001f));
    // }

    // [Burley, 2012, "Physically Based Shading at Disney", Course Notes, v3]
    inline __both__ vec2f roughnessToAlpha(float roughness, float anisotropy=0.f)
    {
      float aspect = sqrt(1.f - 0.9f * anisotropy);
      return vec2f(max(sqr(roughness) / aspect, 0.001f),
                        max(sqr(roughness) * aspect, 0.001f));
    }
    

    
    struct GGXDistribution {
      inline __both__ GGXDistribution(vec2f alpha)
        : alpha(alpha)
      {}

      inline __both__ float eval(const vec3f& wh) const
      {
        float cosTheta = wh.z;
        float cosTheta2 = sqr(cosTheta);
        
        float e = (sqr(wh.x / this->alpha.x) + sqr(wh.y / this->alpha.y)) / cosTheta2;
        return rcp(pi * this->alpha.x * this->alpha.y * sqr(cosTheta2 * (1.f + e)));
      }

      inline __both__
      float evalLambda(const vec3f& wo) const
      {
        float cosThetaO = wo.z;
        float cosThetaO2 = sqr(cosThetaO);
        float invA2 = (sqr(wo.x * this->alpha.x) + sqr(wo.y * this->alpha.y)) / cosThetaO2;
        return 0.5f * (-1.f + sqrt(1.f+invA2));
      }
      

      inline __both__
      float evalG1(const vec3f& wo, float cosThetaOH) const
      {
        float cosThetaO = wo.z;
        if (cosThetaO * cosThetaOH <= 0.f)
          return 0.f;
        
        return rcp(1.f + evalLambda(wo));
      }

      inline __both__
      float evalG2(const vec3f& wo,
                   const vec3f& wi,
                   float cosThetaOH,
                   float cosThetaIH) const
      {
        float cosThetaO = wo.z;
        float cosThetaI = wi.z;
        if (cosThetaO * cosThetaOH <= 0.f || cosThetaI * cosThetaIH <= 0.f)
          return 0.f;
        
        return rcp(1.f + this->evalLambda(wo) + this->evalLambda(wi));
      }


      // Fast visible normal sampling (wo must be in the upper hemisphere)
      // [Heitz, 2017, "A Simpler and Exact Sampling Routine for the GGX Distribution of Visible Normals"]
      inline __both__ vec3f sampleVisible(const vec3f& wo,
                                            float& pdf,
                                            const vec2f& s) const
      {
        // Stretch wo
        vec3f V = normalize(vec3f(this->alpha.x * wo.x, this->alpha.y * wo.y, wo.z));

        // Orthonormal basis
        vec3f T1 = (V.z < 0.9999f) ? normalize(cross(V, vec3f(0,0,1))) : vec3f(1,0,0);
        vec3f T2 = cross(T1, V);

        // Sample point with polar coordinates (r, phi)
        float a = 1.f / (1.f + V.z);
        float r = sqrt(s.x);
        float phi = (s.y<a) ? s.y/a * pi : pi + (s.y-a)/(1.f-a) * pi;
        float P1 = r*cos(phi);
        float P2 = r*sin(phi)*((s.y<a) ? 1.f : V.z);

        // Compute normal
        vec3f wh = P1*T1 + P2*T2 + sqrt(max(0.f, 1.f - P1*P1 - P2*P2))*V;

        // Unstretch
        wh = normalize(vec3f(this->alpha.x * wh.x, this->alpha.y * wh.y, max(0.f, wh.z)));

        // Compute pdf
        float cosThetaO = wo.z;
        pdf = this->evalG1(wo, dot(wo, wh)) * abs(dot(wo, wh)) * this->eval(wh) / abs(cosThetaO);
        return wh;
      }

      inline __both__
      float evalVisible(const vec3f& wh,
                        const vec3f& wo,
                        float cosThetaOH,
                        float& pdf) const
      {
        float cosThetaO = wo.z;
        float D = this->eval(wh);
        pdf = this->evalG1(wo, cosThetaOH) * abs(cosThetaOH) * D / abs(cosThetaO);
        return D;
      }
      

      
      vec2f alpha;
    };
    // struct GGXDistribution1 {
    //   inline __both__ GGXDistribution1(float a) : alpha(a) {}
    //   inline __both__ void init(float alpha) { this->alpha = alpha; }
    //   inline __both__
    //   float evalLambda(const vec3f& wo) const
    //   {
    //     float cosThetaO = wo.z;
    //     float cosThetaO2 = sqr(cosThetaO);
    //     float invA2 = (sqr(wo.x * (float)alpha) + sqr(wo.y * (float)alpha)) / cosThetaO2;
    //     return 0.5f * (-1.f + sqrt(1.f+invA2));
    //   }
    //   inline __both__
    //   float evalVisible(const vec3f& wh, const vec3f& wo,
    //                                       float cosThetaOH, float& pdf) const
    //   {
    //     float cosThetaO = wo.z;
    //     float D = eval(wh);
    //     pdf = evalG1(wo, cosThetaOH) * abs(cosThetaOH) * D / abs(cosThetaO);
    //     return D;
    //   }
    //   inline __both__
    //   float evalG1(const vec3f& wo, float cosThetaOH) const
    //   {
    //     float cosThetaO = wo.z;
    //     if (cosThetaO * cosThetaOH <= 0.f)
    //       return 0.f;
        
    //     return rcp(1.f + evalLambda(wo));
    //   }
    //   inline __both__
    //   float evalG2(const vec3f& wo, const vec3f& wi, float cosThetaOH, float cosThetaIH) const
    //   {
    //     float cosThetaO = wo.z;
    //     float cosThetaI = wi.z;
    //     if (cosThetaO * cosThetaOH <= 0.f || cosThetaI * cosThetaIH <= 0.f)
    //       return 0.f;
        
    //     return rcp(1.f + evalLambda(wo) + evalLambda(wi));
    //   }
    //   inline __both__
    //   float eval(const vec3f& wh) const
    //   {
    //     float cosTheta = wh.z;
    //     float cosTheta2 = sqr(cosTheta);
        
    //     // float e = (sqr(wh.x / self.alpha.x) + sqr(wh.y / self.alpha.y)) / cosTheta2;
    //     float e = (sqr(wh.x / (float)alpha) + sqr(wh.y / (float)alpha)) / cosTheta2;
    //     // return rcp(pi * self.alpha.x * self.alpha.y * sqr(cosTheta2 * (1.f + e)));
    //     return rcp(pi * (float)alpha * (float)alpha * sqr(cosTheta2 * (1.f + e)));
    //   }

    //   // Fast visible normal sampling (wo must be in the upper hemisphere)
    //   // [Heitz, 2017, "A Simpler and Exact Sampling Routine for the GGX Distribution of Visible Normals"]
    //   inline __both__
    //   vec3f sampleVisible(const vec3f& wo, float& pdf, const vec2f& s) const
    //   {
    //     // Stretch wo
    //     vec3f V = normalize(vec3f((float)this->alpha * wo.x, 0.f, wo.z));

    //     // Orthonormal basis
    //     vec3f T1 = (V.z < 0.9999f) ? normalize(cross(V, vec3f(0,0,1))) : vec3f(1,0,0);
    //     vec3f T2 = cross(T1, V);

    //     // Sample point with polar coordinates (r, phi)
    //     float a = 1.f / (1.f + V.z);
    //     float r = sqrtf(s.x);
    //     float phi = (s.y<a) ? s.y/a * pi : pi + (s.y-a)/(1.f-a) * pi;
    //     float P1 = r*cosf(phi);
    //     float P2 = r*sinf(phi)*((s.y<a) ? 1.f : V.z);

    //     // Compute normal
    //     vec3f wh = P1*T1 + P2*T2 + sqrtf(max(0.f, 1.f - P1*P1 - P2*P2))*V;

    //     // Unstretch
    //     wh = normalize(vec3f((float)this->alpha/* .x */ * wh.x,
    //                          0.f /*this->alpha.y* wh.y */,
    //                          max(0.f, wh.z)));

    //     // Compute pdf
    //     float cosThetaO = wo.z;
    //     pdf = evalG1(wo, dot(wo, wh)) * fabsf(dot(wo, wh)) * eval(wh) / fabsf(cosThetaO);
    //     return wh;
    //   }

      
    //   half alpha;
    // };
    
  }
}
