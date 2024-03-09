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
#include "barney/material/bsdfs/DielectricLayer.h"
#include "barney/material/bsdfs/MicrofacetConductor.h"
#include "barney/material/bsdfs/MultiBSDF.h"

namespace barney {
  namespace render {

    struct MetallicPaint {
      // struct Substrate {
      //   inline __device__
      //   vec3f getAlbedo(bool dbg=false) const
      //   { return vec3f(0.f); }
        
      //   inline __device__
      //   EvalRes eval(render::DG dg, vec3f wi, bool dbg=false) const
      //   {
      //     return EvalRes::zero();
      //     // EvalRes lambert_eval = lambert.eval(dg,wi,dbg);
      //     // float   lambert_imp  = lambert.importance();
      //     // EvalRes facets_eval = facets.eval(dg,wi,dbg);
      //     // float   facets_imp  = facets.importance();
      //     // // EvalRes velvety_eval  = velvety.eval(dg,wi,dbg);
      //     // // float   velvety_imp   = minneart.importance();
      //     // EvalRes our_eval;
      //     // our_eval.value
      //     //   = lambert_eval.value
      //     //   + facets_eval.value
      //     //   ;
      //     // // if (dbg) {
      //     // //   printf("lambert %f %f %f\n",
      //     // //          lambert_eval.value.x,
      //     // //          lambert_eval.value.y,
      //     // //          lambert_eval.value.z);
      //     // //   printf("facets %f %f %f\n",
      //     // //          facets_eval.value.x,
      //     // //          facets_eval.value.y,
      //     // //          facets_eval.value.z);
      //     // // }
      //     // our_eval.pdf 
      //     //   = (lambert_imp*lambert_eval.pdf+facets_imp*facets_eval.pdf)
      //     //   / max(1e-20f,lambert_imp+facets_imp);
      //     // return our_eval;
      //   }
        
        // Lambert lambert;
        // MicrofacetConductor<FresnelSchlick1> facets;
      //   enum { bsdfType = Lambert::bsdfType };
      // };
      
      struct HitBSDF {
        inline __device__
        EvalRes eval(render::DG dg, vec3f wi, bool dbg=false) const
        {
          const vec3f r = this->flakeColor;
          const float g = this->flakeAmount;
          FresnelSchlick1 fresnel(r,g);
          MultiBSDF2<Lambert, MicrofacetConductor<FresnelSchlick1>>
            substrate(Lambert(vec3f(baseColor)),
                      MicrofacetConductor<FresnelSchlick1>(fresnel,this->flakeSpread));
          DielectricLayer1<MultiBSDF2<Lambert, MicrofacetConductor<FresnelSchlick1>>>
            bsdf(substrate,eta);
          return bsdf.eval(dg,wi,dbg);
          // return dielectricLayer.eval(dg,wi,dbg);
        }
        
        inline __device__       
        vec3f getAlbedo(bool dbg=false) const
        {
          return vec3f(0.f);
          // return dielectricLayer.getAlbedo(dbg);
        }
        
        vec3h baseColor;
        half flakeAmount;
        vec3h flakeColor;
        half flakeSpread;
        half eta;
        
        // DielectricLayer1<Substrate> dielectricLayer;
        enum { bsdfType = BSDF_SPECULAR_REFLECTION
          // | Substrate::bsdfType
        };
      };
      struct DD {
        inline __device__
        void make(HitBSDF &multi, bool dbg) const
        {
          
           // if (self->flakeAmount > 0.f) {
          //   const vec3f r = self->flakeColor;
          //   const vec3f g = make_vec3f(self->flakeAmount);
          //   Fresnel *uniform fresnel = FresnelSchlick_create(ctx, r, g);
          //   MultiBSDF_add(bsdf,
          //     MicrofacetConductor_create(ctx, shadingFrame, fresnel, self->flakeSpread, 0.f), 1.f, luminance(r));
          // }
          // float flakeAmount = max(.1f,this->flakeAmount);
          // vec3f r = flakeColor;
          // float g = flakeAmount;
          // FresnelSchlick1 fresnel; fresnel.init(r,g);
          // multi.dielectricLayer.substrate.facets.init(fresnel,flakeSpread);
          // multi.dielectricLayer.substrate.lambert.init(baseColor,dbg);
          // multi.dielectricLayer.init(this->eta);
          multi.baseColor = baseColor;
          multi.flakeAmount = flakeAmount;
          multi.flakeColor = flakeColor;
          multi.flakeSpread = flakeSpread;
          multi.eta = eta;
        }
        vec3f baseColor;
        float flakeAmount;
        vec3f flakeColor;
        float flakeSpread;
        float eta;
      };
    };
    
  }
}
