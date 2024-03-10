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

namespace barney {
  namespace render {

    /*! dielectirclayer, but for metallicpaint, where transmittance, thickness, and wight are 1 */
    template<typename Substrate>
    struct DielectricLayer1 : public BSDF {
      inline __device__
      DielectricLayer1(const Substrate &substrate, float _eta)
        : BSDF(.5f),
          substrate(substrate),
          eta((_eta <= 1.f) ? _eta : rcp(_eta))
      {}
      // inline __device__
      // void init(float eta) {
      //   this->eta = (eta <= 1.f) ? eta : rcp(eta);
      // }

      // inline __device__       
      // vec3f getAlbedo(bool dbg=false) const
      // { return vec3f(.5f); }
      
      inline __device__
      EvalRes eval(render::DG dg, vec3f wi, bool dbg=false) const
      { 
        // inline BSDF_EvalRes DielectricLayer_eval(const varying BSDF* uniform super,
        //                                          const vec3f& wo, const vec3f& wi)
        // {
        // const varying DielectricLayer* uniform self = (const varying DielectricLayer* uniform)super;

        // float cosThetaO = dot(wo, getN(super));
        const vec3f N = dg.Ns;
        const vec3f wo = dg.wo;
        float cosThetaO = dot(wo, N);
        // if (cosThetaO <= 0.f)
        //   return make_BSDF_EvalRes_zero();
        if (cosThetaO <= 0.f)
          return EvalRes::zero();
        // float cosThetaI = dot(wi, getN(super));
        float cosThetaI = dot(wi, N);

        // Fresnel term
        float cosThetaO1; // positive
        // float Fo = fresnelDielectricEx(cosThetaO, cosThetaO1, self->eta) * self->weight;
        const float self_weight = 1.f;
        float Fo = fresnelDielectricEx(cosThetaO, cosThetaO1, (float)this->eta) * self_weight;

        // Evaluate the substrate
        // Ignore refraction
        // BSDF_EvalRes substrate;
        // foreach_unique (f in self->substrate)
        //   substrate = f->eval(f, wo, wi);
        EvalRes substrate = this->substrate.eval(dg,wi,dbg);

        float cosThetaI1; // positive
        // float Fi = fresnelDielectricEx(abs(cosThetaI), cosThetaI1, self->eta) * self->weight;
        float Fi = fresnelDielectricEx(fabsf(cosThetaI), cosThetaI1, this->eta) * self_weight;
        
        // Apply the coating medium absorption
        // Use refracted angles for computing the absorption path length
        float lengthO1 = rcp(cosThetaO1);
        float lengthI1 = rcp(cosThetaI1);
        float length = lengthO1 + lengthI1;
        if (cosThetaI <= 0.f) length *= 0.5f; // for transmission, use the average length
        // substrate.value
        //   = lerp(self->weight, substrate.value,
        //          substrate.value * pow(self->transmittance, self->thickness * length));
        // substrate.value
        //   = lerp(vec3f(self_weight), substrate.value,
        //          substrate.value * pow(this->transmittance, (float)this->thickness * length));
        // iw : self.weight is 1.f, and so is transmittance .. !?

#if 1
        float T = min(1.f - Fo, 1.f - Fi);
        substrate.value = substrate.value * T;
#else
        // Energy conservation
        float T;
        if (self->substrate->type & ~BSDF_DIFFUSE)
          T = min(1.f - Fo, 1.f - Fi); // for generic (non-diffuse) substrates [Kulla and Conty, 2017]
        else
          T = (1.f - Fo) * (1.f - Fi) * rcp(1.f - self->Favg); // for diffuse substrates [Kelemen and Szirmay-Kalos, 2001]
        substrate.value = substrate.value * T;
#endif
        
        substrate.pdf *= (1.f - Fo);
        return substrate;
      }

      
      half eta;
      Substrate substrate;
      // vec3h transmittance;
      // half thickness;
      // half Favg;
      // half weight;
    };

  }
}
