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

    template<typename A, typename B>
    struct MultiBSDF2 {
      inline __both__ MultiBSDF2(A a, B b)
        : a(a), b(b)
      {}

      inline __device__
      vec3f getAlbedo(bool dbg=false) const {
        return a.getAlbedo(dbg)+b.getAlbedo(dbg);
      }

      inline __device__
      EvalRes eval(render::DG dg, vec3f wi, bool dbg=false) const
      {
        EvalRes a_eval = a.eval(dg,wi,dbg);
        float   a_imp  = a.importance();
        EvalRes b_eval  = b.eval(dg,wi,dbg);
        float   b_imp   = b.importance();
        EvalRes our_eval;
        our_eval.value = a_eval.value + b_eval.value;
        our_eval.pdf
          = (a_imp*a_eval.pdf+b_imp*b_eval.pdf)
          / max(1e-20f,a_imp+b_imp);
        return our_eval;
      }

      enum { bsdfType = A::bsdfType | B::bsdfType };
      
      A a;
      B b;
    };
      
  }
}
