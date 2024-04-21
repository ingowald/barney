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

#pragma once

#include "barney/material/device/DG.h"
#include "barney/material/device/BSDF.h"
#include "barney/material/bsdfs/MicrofacetConductor.h"
#include "barney/material/bsdfs/Fresnel.h"

namespace barney {
  namespace render {

    struct Metal {
      struct HitBSDF {
        inline __device__
        vec3f getAlbedo(bool dbg=false) const {
          // return (vec3f)lambert.albedo;
          return vec3f(0.f);
        }
        
        inline __device__
        SampleRes sample(const DG &dg,
                         Random &randomF,
                         bool dbg = false)
        {
          FresnelConductorRGBUniform fresnel((vec3f)eta,(vec3f)k);
          MicrofacetConductor<FresnelConductorRGBUniform> facets(fresnel,roughness);
          return facets.sample(dg,randomF,dbg);
        }
        
        inline __device__
        EvalRes eval(render::DG dg, vec3f wi, bool dbg=false) const
        {
          FresnelConductorRGBUniform fresnel((vec3f)eta,(vec3f)k);
          MicrofacetConductor<FresnelConductorRGBUniform> facets(fresnel,roughness);
          return facets.eval(dg,wi,dbg);
        }
        
        vec3h eta;
        vec3h k;
        half roughness;

        enum { bsdfType = MicrofacetConductor<FresnelConductorRGBUniform>::bsdfType };
      };
      struct DD {
        inline __device__
        void make(HitBSDF &multi, bool dbg) const
        {
          multi.eta = eta;
          multi.k = k;
          multi.roughness = roughness;
        }
        vec3f eta;
        vec3f k;
        float roughness;
      };
    };
    
  }
}
