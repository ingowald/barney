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
#include "barney/material/bsdfs/RobustDielectric.h"

namespace barney {
  namespace render {

    struct Medium {
      // inline __device__ Medium() {}
      vec3h attenuation;
      float ior;
    };
    
    struct Glass {
      struct HitBSDF {
        inline __device__ HitBSDF() {}
        
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
          float eta
            = (!dg.insideMedium)
            ? mediumOutside.ior / mediumInside.ior
            : mediumInside.ior / mediumOutside.ior;
          // if (dbg) printf("eta is %f\n",eta);
          return RobustDielectric(eta).sample(dg,randomF,dbg);
        }
        
        inline __device__
        EvalRes eval(render::DG dg, vec3f wi, bool dbg=false) const
        {
          return EvalRes::zero();
        }

        inline __device__
        vec3f getTransparency(// const Medium& currentMedium
                              ) const
        {
          // float eta
          //   = (currentMedium == mediumOutside)
          //   ? mediumOutside.ior*rcp(mediumInside.ior)
          //   : self->mediumInside.ior*rcp(self->mediumOutside.ior);
          // float eta = mediumOutside.ior/mediumInside.ior;
          // float cosThetaO = max(-dot(ray.dir, dg.Ns), 0.0f);
          // return make_vec3f(1.0f-fresnelDielectric(cosThetaO, eta));
          return vec3f(1.f);
        }

        Medium mediumInside;
        Medium mediumOutside;
        
        enum { bsdfType = RobustDielectric::bsdfType };
      };
      struct DD {
        inline __device__
        void make(HitBSDF &multi, bool dbg) const
        {
          multi.mediumInside  = mediumInside;
          multi.mediumOutside = mediumOutside;
        }
        // vec3f transmission;
        Medium mediumInside;
        Medium mediumOutside;
      };
    };
    
  }
}
