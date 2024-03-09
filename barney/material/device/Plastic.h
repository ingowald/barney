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
#include "barney/material/bsdfs/MicrofacetDielectricLayer.h"

namespace barney {
  namespace render {

    typedef uint32_t BSDFType;

    template<typename Substrate>
    struct DielectricLayerT {
      inline __device__
      DielectricLayerT(float eta, Substrate substrate) : eta(eta), substrate(substrate) {}
      
      inline __device__
      EvalRes eval(render::DG dg, vec3f wi, bool dbg=false) const
      { return EvalRes::zero(); }
        
      Substrate substrate;
      float eta;
    };

    struct Plastic {
      struct HitBSDF {
        inline __device__
        vec3f getAlbedo(bool dbg=false) const { return vec3f(0.f); }
        
        inline __device__
        EvalRes eval(const Globals::DD &globals,
                     render::DG dg, vec3f wi, bool dbg=false) const
        {
          if ((float)roughness == 0.f) {
            return
              DielectricLayerT<Lambert>((float)eta,
                                        Lambert::create((vec3f)pigmentColor))
              .eval(dg,wi,dbg);
          } else {
            return
              MicrofacetDielectricLayer<Lambert>((float)eta,(float)roughness,
                                                 Lambert::create((vec3f)pigmentColor))
              .eval(globals,dg,wi,dbg);
          }
        }
        
        vec3h pigmentColor;
        half eta;
        half roughness;
        
        enum { bsdfType = BSDF_DIFFUSE_REFLECTION };
      };
      struct DD {
        inline __device__
        void make(HitBSDF &multi, bool dbg) const
        {
          multi.eta = eta;
          multi.roughness = roughness;
          multi.pigmentColor = pigmentColor;
        }
        vec3f pigmentColor;
        float eta;
        float roughness;
      };
    };
    
  }
}
