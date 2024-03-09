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
#include "barney/material/bsdfs/GGXDistribution.h"
#include "barney/material/bsdfs/optics.h"
#include "barney/material/bsdfs/MicrofacetAlbedo.h"

namespace barney {
  namespace render {

    template<typename Substrate>
    struct MicrofacetDielectricLayer {
      // float eta, vec3f transmittance, float thickness, float roughness, float anisotropy,
      // float weight)
      inline __device__
      MicrofacetDielectricLayer(float eta, float roughness,
                                Substrate substrate)
        : eta(eta <= 1.f ? eta : rcpf(eta)),
          roughness(roughness), substrate(substrate),
          microfacet(roughnessToAlpha(roughness))
      {}
      
      inline __device__
      EvalRes eval(const Globals::DD &globals, render::DG dg, vec3f wi, bool dbg=false) const
      {
        vec3f wo = dg.wo;
        vec3f N = dg.N;
        // float cosThetaO = dot(wo, getN(super));
        float cosThetaO = dot(wo, N);
        if (cosThetaO <= 0.f)
          // return make_BSDF_EvalRes_zero();
          return EvalRes::zero();
        // float cosThetaI = dot(wi, getN(super));
        float cosThetaI = dot(wi, N);

        // Evaluate the substrate
        // Ignore refraction
        // BSDF_EvalRes substrate;
        // foreach_unique (f in self->substrate)
        //   substrate = f->eval(f, wo, wi);
        EvalRes substrate = this->substrate.eval(dg,wi,dbg);
        
        // Apply the coating medium absorption
        // Use refracted angles for computing the absorption path length
        // float lengthO1 = rcp(refract(cosThetaO, self->eta)); // rcp(cosThetaO1)
        float lengthO1 = rcp(refract(cosThetaO, this->eta)); // rcp(cosThetaO1)
        // float lengthI1 = rcp(refract(abs(cosThetaI), self->eta)); // rcp(cosThetaI1)
        float lengthI1 = rcp(refract(abs(cosThetaI), this->eta)); // rcp(cosThetaI1)
        float length = lengthO1 + lengthI1;
        if (cosThetaI <= 0.f) length *= 0.5f; // for transmission, use the average length
        // substrate.value = lerp(self->weight, substrate.value, substrate.value * pow(self->transmittance, self->thickness * length));
#if 0
        // iw - weight and transmittance are both 1.f, so this evalautes to 1
        substrate.value
          = lerp(self->weight, substrate.value, substrate.value * pow(self->transmittance, self->thickness * length));
#endif
        
        // Energy conservation
        // float Eo = MicrofacetAlbedo_eval(cosThetaO, this->roughness);
        // float Ei = MicrofacetAlbedo_eval(fabsf(cosThetaI), this->roughness);
        float Ro = MicrofacetDielectricReflectionAlbedoTable_eval
          (globals, cosThetaO, this->eta, this->roughness)
          // + self->fmsScale * (1.f - Eo)) * self->weight // add missing energy
          ;
        float Ri = MicrofacetDielectricReflectionAlbedoTable_eval
          (globals, fabsf(cosThetaI), this->eta, this->roughness)
          // + self->fmsScale * (1.f - Ei)) * self->weight; // add missing energy
          ;
        float T;
#if 1
        if (this->substrate.bsdfType & ~BSDF_DIFFUSE)
          T = min(1.f - Ro, 1.f - Ri); // for generic (non-diffuse) substrates [Kulla and Conty, 2017]
        else
          T = 1.f;
#else
        if (self->substrate->type & ~BSDF_DIFFUSE)
          T = min(1.f - Ro, 1.f - Ri); // for generic (non-diffuse) substrates [Kulla and Conty, 2017]
        else
          T = (1.f - Ro) * (1.f - Ri) * rcp(1.f - self->Ravg); // for diffuse substrates [Kelemen and Szirmay-Kalos, 2001]
#endif
        substrate.value = substrate.value * T;
        
        float coatPickProb = Ro;
        float substratePickProb = 1.f - coatPickProb;
        
        if (cosThetaI > 0.f)
          {
            // Compute the microfacet normal
            vec3f wh = normalize(wo + wi);
            float cosThetaOH = dot(wo, wh);

            // Fresnel term
            // float F = fresnelDielectric(cosThetaOH, self->eta) * self->weight;
            float F = fresnelDielectric(cosThetaOH, this->eta) * weight();

            // Evaluate the coating reflection
            // float cosThetaH = dot(wh, getN(super));
            float cosThetaH = dot(wh, N);
            float cosThetaIH = dot(wi, wh);

            // linear3f toLocal = transposed(getFrame(super));
            linear3f localFrame = owl::common::frame(dg.N);
            linear3f toLocal = localFrame.transposed();
            vec3f wo0 = toLocal * wo;
            vec3f wi0 = toLocal * wi;
            vec3f wh0 = toLocal * wh;

            float whPdf;
            // float D = evalVisible(self->microfacet, wh0, wo0, cosThetaOH, whPdf);
            // float G = evalG2(self->microfacet, wo0, wi0, cosThetaOH, cosThetaIH);
            float D = microfacet.evalVisible(wh0, wo0, cosThetaOH, whPdf);
            float G = microfacet.evalG2(wo0, wi0, cosThetaOH, cosThetaIH);

            // Energy compensation
            // float fms = self->fmsScale * ((1.f - Eo) * (1.f - Ei) * rcp(pi * (1.f - self->Eavg)) * cosThetaI);

            // BSDF_EvalRes coat;
            EvalRes coat;
            coat.pdf = whPdf * rcp(4.f*cosThetaOH);
            // coat.value = make_vec3f(F * D * G * rcp(4.f*cosThetaO) + fms);
            coat.value = vec3f(F * D * G * rcp(4.f*cosThetaO)/* + fms*/);

            // Compute the total reflection
            // BSDF_EvalRes res;
            EvalRes res;
            res.pdf = coatPickProb * coat.pdf + substratePickProb * substrate.pdf;
            res.value = coat.value + substrate.value;
            return res;
          }
        else
          {
            // Return the substrate transmission
            substrate.pdf *= substratePickProb;
            return substrate;
          }
      }
      
      // vec3f transmittance; --> assumed 1.f
      inline __device__ vec3f transmittance() const { return vec3f(1.f); }
      inline __device__ float thickness() const { return 1.f; }
      // float thickness;
      GGXDistribution1 microfacet;
      float roughness;
      float eta;
      Substrate substrate;
                     inline __device__ float weight() const { return 1.f; }
      // float weight; --> assumed 1.f
                     };
    
  }
}
