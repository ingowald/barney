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
#include "barney/material/bsdfs/Lambert.h"
#include "barney/material/bsdfs/optics.h"
#include "barney/material/bsdfs/GGXDistribution.h"

namespace barney {
  namespace render {

    template<typename Fresnel>
    struct MicrofacetConductor {
      enum { bsdfType = BSDF_GLOSSY_REFLECTION };
      inline __device__
      float importance() const { return luminance(fresnel.r); }
      
      inline __device__
      EvalRes eval(DG dg, vec3f wi, bool dbg = false) const
      {
        linear3f localFrame = owl::common::frame(dg.N);
        
        vec3f wo = dg.wo;
        float cosThetaO = dot(wo, dg.N);
        float cosThetaI = dot(wi, dg.N);
        if (cosThetaO <= 0.f || cosThetaI <= 0.f) {
          return EvalRes(vec3f(0.f),0.f);
        }
        
        EvalRes res;
        // Compute the microfacet normal
        vec3f wh = normalize(wi + wo);
        float cosThetaH = dot(wh, dg.N);
        float cosThetaOH = dot(wo, wh);
        float cosThetaIH = dot(wi, wh);
        
        // linear3f toLocal = transposed(getFrame(super));
        linear3f toLocal = localFrame.transposed();
        vec3f wo0 = toLocal * wo;
        vec3f wi0 = toLocal * wi;
        vec3f wh0 = toLocal * wh;
        
        vec3f F = fresnel.eval(cosThetaOH,dbg);
        float whPdf;
        float D = microfacet.evalVisible(wh0, wo0, cosThetaOH, whPdf);
        float G = microfacet.evalG2(wo0, wi0, cosThetaOH, cosThetaIH);

        // // Energy compensation
        // float Eo = MicrofacetAlbedoTable_eval(cosThetaO, roughness);
        // float Ei = MicrofacetAlbedoTable_eval(cosThetaI, roughness);
        // vec3f fms = self->fmsScale * ((1.f - Eo) * (1.f - Ei) * rcp(pi * (1.f - self->Eavg)) * cosThetaI);

        // float Eo = MicrofacetAlbedo_integrate(cosThetaO, roughness);
        // float Ei = MicrofacetAlbedo_integrate(cosThetaO, roughness);
        // }
        res.pdf = whPdf * rcp(4.f*abs(cosThetaOH));
        res.value = F * (D * G * rcp(4.f*cosThetaO));// + fms;
        
        return res;
      }
      
      inline __device__
      MicrofacetConductor(Fresnel fresnel, float roughness)
        : fresnel(fresnel),
          roughness(roughness),
          microfacet(roughnessToAlpha(roughness))
      {}
      
      // inline __device__
      // void init(Fresnel fresnel, float roughness)
      // {
      //   // , float anisotropy --> 0.f
      //   // const float anisotropy = 0.f;
        
      //   // self->Eavg = MicrofacetAlbedoTable_evalAvg(roughness);
      //   // Eavg = MicrofacetAlbedoTable::evalAvg(roughness);
      //   // // vec3f Favg = fresnel->evalAvg(fresnel);
      //   // Favg = fresnel.evalAvg();
      //   // // self->fmsScale = sqr(Favg) * self->Eavg / (1.f - Favg * (1.f - self->Eavg)); // Stephen Hill's tweak
      //   // fmsScale = sqr(Favg)*Eavg / (1.f - Favg * (1.f-Eavg));
        
      //   // BSDF_Constructor(&self->super, Favg * self->Eavg, // TODO better estimate
      //   //                  BSDF_GLOSSY_REFLECTION,
      //   //                  MicrofacetConductor_eval, MicrofacetConductor_sample,
      //   //                  frame);
      //   // self->fresnel = fresnel;
      //   this->fresnel = fresnel;
      //   // self->microfacet = make_GGXDistribution(roughnessToAlpha(roughness, anisotropy));
      //   // microfacet.init(roughnessToAlpha(roughness,anisotropy));
      //   microfacet.init(roughnessToAlpha(roughness));
      //   // self->roughness = roughness;
      //   this->roughness = roughness;
      // }
      
      Fresnel fresnel;
      GGXDistribution microfacet;
      half roughness;
      
      // // Energy compensation [Kulla and Conty, 2017]
      // half Eavg;
      // vec3h fmsScale;
    };


    

  }
}
