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

#include "barney/packedBSDF/fromOSPRay/BSDF.h"

namespace BARNEY_NS {
  namespace render {

    typedef uint32_t BSDFType;

    struct Lambert : public BSDF {

      inline __both__
      Lambert(vec3f R, bool dbg = false)
        : BSDF(R)
      {}
      inline Lambert(const Lambert &) = default;
      // { Lambert l; l.init(R); return l; }

      // static inline __both__
      // Lambert create(vec3f R, bool dbg = false)
      // { Lambert l; l.init(R); return l; }
                     
      // inline __both__ void init(vec3f R, bool dbg = false)
      // { BSDF::init(R); }
        
      inline __both__
      EvalRes eval(DG dg, vec3f wi, bool dbg = false) const
      {
        EvalRes res;
        float cosThetaI = max(dot(wi, dg.Ns), 0.f);
        res.pdf = cosineSampleHemispherePDF(cosThetaI);
        res.value = (vec3f)albedo * ONE_OVER_PI * cosThetaI;
        return res;
      }
      
      enum { bsdfType = BSDF_DIFFUSE_REFLECTION };
    };
    
  }
}
