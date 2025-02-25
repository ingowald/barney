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

#include "barney/render/DG.h"
#include "barney/render/HitAttributes.h"
#include "barney/packedBSDF/fromOSPRay/RobustDielectric.h"

namespace BARNEY_NS {
  namespace render {
    namespace packedBSDF {
      
      struct Glass {
        inline __rtc_device
        vec3f getAlbedo(bool dbg) const;
        
        inline __rtc_device
        float getOpacity(bool isShadowRay,
                              bool isInMedium,
                              vec3f rayDir,
                              vec3f Ng,
                              bool dbg=false) const;
        inline __rtc_device float pdf(DG dg, vec3f wi, bool dbg) const;
        inline __rtc_device EvalRes eval(DG dg, vec3f wi, bool dbg) const;
        inline __rtc_device void scatter(ScatterResult &scatter,
                                       const render::DG &dg,
                                       Random &random,
                                       bool dbg) const;

        float  ior;
        float3 attenuation;
      };

      inline __rtc_device EvalRes Glass::eval(DG dg, vec3f wi, bool dbg) const
      {
        return EvalRes::zero();
      }

      
      inline __rtc_device
      float Glass::getOpacity(bool isShadowRay,
                              bool isInMedium,
                              vec3f rayDir,
                              vec3f Ng,
                              bool dbg) const
      {
        if (isShadowRay) {
          // if (dbg) printf(" glass on shadow ray -> opacity := 0\n");
          return 0.f;
        }

#if 0
        bool isEntering = dot(rayDir,Ng) < 0.f;
        if (dbg) {
          printf("Glass::getOpacity: dir %f %f %f Ng %f %f %f, inMedium %i, entering %i\n",
                 rayDir.x,
                 rayDir.y,
                 rayDir.z,
                 Ng.x,
                 Ng.y,
                 Ng.z,
                 (int)isInMedium,
                 (int)isEntering);
        }
        if (isEntering && isInMedium ||
            !isEntering && !isInMedium)
          return 0.f;
#endif   
        return 1.f;
      }
      
      inline __rtc_device
      void Glass::scatter(ScatterResult &scatter,
                          const render::DG &dg,
                          Random &random,
                          bool dbg) const
      {
        float eta
          = (!dg.insideMedium)
          ? 1.f/ior
          : ior;
        // ? mediumOutside.ior / mediumInside.ior
        //     : mediumInside.ior / mediumOutside.ior;
        // if (dbg) printf("eta is %f\n",eta);
        SampleRes sampleRes = RobustDielectric(eta).sample(dg,random,dbg);
        scatter.f_r
          = sampleRes.weight;
        scatter.dir
          = sampleRes.wi;
        scatter.pdf
          = sampleRes.pdf;
        scatter.offsetDirection
          = (sampleRes.type == BSDF_SPECULAR_TRANSMISSION)
          ? -1.f
          : +1.f;
        scatter.changedMedium
          = (sampleRes.type == BSDF_SPECULAR_TRANSMISSION);
        // if (dbg) printf("glass f_r %f %f %f pdf %f\n",
        //                 scatter.f_r.x,
        //                 scatter.f_r.y,
        //                 scatter.f_r.z,
        //                 scatter.pdf);
      }
      
      
      inline __rtc_device float Glass::pdf(DG dg, vec3f wi, bool dbg) const
      {
        return 0.f;
      }
      
    }
  }
}

