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

#include "barney/material/math.h"
#include "barney/material/Globals.h"
#include "Fresnel.h"
#include "GGXDistribution.h"

#define MICROFACET_ALBEDO_TABLE_SIZE 32
#define MICROFACET_DIELECTRIC_ALBEDO_TABLE_SIZE 16
#define MICROFACET_DIELECTRIC_ALBEDO_TABLE_MIN_IOR 1.f
#define MICROFACET_DIELECTRIC_ALBEDO_TABLE_MAX_IOR 3.f

namespace barney {
  namespace render {

    inline __both__
    float MicrofacetDielectricReflectionAlbedo_sample(float cosThetaO,
                                                      float eta,
                                                      const GGXDistribution1 &microfacet,
                                                      const vec2f &s)
    {
      // Handle edge cases
      cosThetaO = max(cosThetaO, 1e-6f);

      // Make an outgoing vector
      vec3f wo = vec3f(cos2sin(cosThetaO), 0.f, cosThetaO);

      // Sample the microfacet normal
      float whPdf;
      vec3f wh = microfacet.sampleVisible(wo, whPdf, s);

      float cosThetaOH = dot(wo, wh);

      // Fresnel term
      float F = fresnelDielectric(cosThetaOH, eta);

      // Sample the reflection
      vec3f wi = reflect(wo, wh, cosThetaOH);
      float cosThetaI = wi.z;
      if (cosThetaI <= 0.f)
        return 0.f;

      float cosThetaIH = dot(wi, wh);
      float G = microfacet.evalG2(wo, wi, cosThetaOH, cosThetaIH);

      return F * (G * rcp_safe(microfacet.evalG1(wo, cosThetaOH)));
    }

    
    inline __both__
    float MicrofacetDielectricReflectionAlbedo_integrate(float cosThetaO,
                                                         float eta,
                                                         float roughness,
                                                         int numSamples = 1024)
    {
      GGXDistribution1 microfacet(roughness);
      
      int n = sqrt((float)numSamples);
      float sum = 0.f;
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          vec2f s = min((vec2f(i, j) + 0.5f) * (1.f/n), vec2f(1.f - 1e-6f));
          sum += MicrofacetDielectricReflectionAlbedo_sample(cosThetaO, eta, microfacet, s);
        }
      }
      
      return min(sum / (n*n), 1.f);
    }


    inline __both__
    float MicrofacetDielectricAlbedo_sample(float cosThetaO,
                                            float eta,
                                            const GGXDistribution1 &microfacet,
                                            const vec2f& s)
    {
      // Handle edge cases
      cosThetaO = max(cosThetaO, 1e-6f);

      // Make an outgoing vector
      vec3f wo = vec3f(cos2sin(cosThetaO), 0.f, cosThetaO);

      // Sample the microfacet normal
      float whPdf;
      vec3f wh = microfacet.sampleVisible(wo, whPdf, s);
      
      float cosThetaOH = dot(wo, wh);

      // Fresnel term
      float cosThetaTH; // positive
      float F = fresnelDielectricEx(cosThetaOH, cosThetaTH, eta);

      float weight = 0.f;

      // Sample the reflection
      vec3f wi = reflect(wo, wh, cosThetaOH);
      float cosThetaI = wi.z;
      if (cosThetaI > 0.f) {
        float cosThetaIH = dot(wi, wh);
        float G = microfacet.evalG2(wo, wi, cosThetaOH, cosThetaIH);
        weight += F * (G * rcp_safe(microfacet.evalG1(wo, cosThetaOH)));
      }
      
      // Sample the transmission
      // cosThetaTH = -cosThetaIH
      wi = refract(wo, wh, cosThetaOH, cosThetaTH, eta);
      cosThetaI = wi.z;
      if (cosThetaI < 0.f) {
        float cosThetaIH = dot(wi, wh);
        float G = microfacet.evalG2(wo, wi, cosThetaOH, cosThetaIH);
        weight += (1.f-F) * (G * rcp_safe(microfacet.evalG1(wo, cosThetaOH)));
      }
      
      return weight;
    }
    
    
    inline __both__
    float MicrofacetDielectricAlbedo_integrate(float cosThetaO,
                                               float eta,
                                               float roughness,
                                               int numSamples = 1024)
    {
      // GGXDistribution microfacet(roughness, 0.f);
      GGXDistribution1 microfacet(roughness);
      
      int n = sqrtf((float)numSamples);
      float sum = 0.f;
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          vec2f s = min((vec2f(i, j) + 0.5f) * (1.f/n), vec2f(1.f - 1e-6f));
          sum += MicrofacetDielectricAlbedo_sample(cosThetaO, eta, microfacet, s);
        }
      }
      return min(sum / (n*n), 1.f);
    }

    
    inline __device__
    float MicrofacetDielectricReflectionAlbedoTable_eval
    (const Globals::DD &globals, float cosThetaO, float eta, float roughness)
    {
      const int size = MICROFACET_DIELECTRIC_ALBEDO_TABLE_SIZE;

      //if (eta <= 1.f)
      {
        const float minEta = rcp(MICROFACET_DIELECTRIC_ALBEDO_TABLE_MAX_IOR);
        const float maxEta = rcp(MICROFACET_DIELECTRIC_ALBEDO_TABLE_MIN_IOR);
        const float etaParam = (eta - minEta) / (maxEta - minEta);
        const vec3f p = vec3f(cosThetaO, etaParam, roughness) * (size-1);
        return interp3DLinear(p, globals.MicrofacetDielectricReflectionAlbedoTable_dir, vec3i(size));
      }
      /*
        else
        {
        const uniform float minEta = MICROFACET_DIELECTRIC_ALBEDO_TABLE_MIN_IOR;
        const uniform float maxEta = MICROFACET_DIELECTRIC_ALBEDO_TABLE_MAX_IOR;
        const float etaParam = (eta - minEta) / (maxEta - minEta);
        const vec3f p = make_vec3f(cosThetaO, etaParam, roughness) * (size-1);
        return interp3DLinear(p, MicrofacetDielectricReflectionRcpEtaAlbedoTable_dir, make_vec3i(size));
        }
      */
    }
  }
}
