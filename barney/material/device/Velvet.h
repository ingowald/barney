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
#include "barney/material/bsdfs/MultiBSDF.h"

namespace barney {
  namespace render {

    typedef uint32_t BSDFType;

    struct Minneart : public BSDF {
      inline __device__ Minneart(vec3f R, float b, bool dbg = false)
        : BSDF(R), b(b)
      {}
      inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg = false) const
      {
        EvalRes res;
        const float cosThetaI = clamp(dot(wi,dg.Ns));
        const float backScatter = powf(clamp(dot(dg.wo,wi)), (float)b);
        res.pdf = cosineSampleHemispherePDF(cosThetaI);
        res.value = (backScatter * cosThetaI * one_over_pi) * (vec3f)albedo;
        return res;
      }
      
      enum { bsdfType = BSDF_DIFFUSE_REFLECTION };
      /*! The amount of backscattering. A value of 0 means lambertian
       *  diffuse, and inf means maximum backscattering. */
      float b;
    };
    
    struct Velvety : public BSDF {
      inline __device__ Velvety(vec3f R, float f, bool dbg = false)
        : BSDF(R), f(f)
      {}
      inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg = false) const
      {
        EvalRes res;
        const float cosThetaO = clamp(dot(dg.wo,dg.Ns));
        const float cosThetaI = clamp(dot(wi,dg.Ns));
        const float sinThetaO = sqrt(1.0f - cosThetaO * cosThetaO);
        const float horizonScatter = pow(sinThetaO, (float)f);
        res.pdf = cosineSampleHemispherePDF(cosThetaI);
        res.value =  (horizonScatter * cosThetaI * one_over_pi) * (vec3f)albedo;
        return res;
      }
      enum { bsdfType = BSDF_DIFFUSE_REFLECTION };
      /*! The falloff of horizon scattering. 0 no falloff,
       *  and inf means maximum falloff. */
      float f;
    };

    
    struct Velvet {
      struct HitBSDF {
        inline __device__
        vec3f getAlbedo(bool dbg=false) const {
          MultiBSDF2<Minneart,Velvety>
            multi(Minneart(reflectance,backScattering,dbg),
                  Velvety(horizonScatteringColor,horizonScatteringFallOff,dbg));
          return multi.getAlbedo(dbg);
          // return (vec3f)minneart.albedo+(vec3f)velvety.albedo;
        }
        
        inline __device__
        EvalRes eval(render::DG dg, vec3f wi, bool dbg=false) const
        {
          MultiBSDF2<Minneart,Velvety>
            multi(Minneart(reflectance,backScattering,dbg),
                  Velvety(horizonScatteringColor,horizonScatteringFallOff,dbg));
          return multi.eval(dg,wi,dbg);
          // multi.minneart.init(reflectance,backScattering,dbg);
          // multi.velvety.init(horizonScatteringColor,horizonScatteringFallOff,dbg);

          // EvalRes minneart_eval = minneart.eval(dg,wi,dbg);
          // float   minneart_imp  = minneart.importance();
          // EvalRes velvety_eval  = velvety.eval(dg,wi,dbg);
          // float   velvety_imp   = minneart.importance();
          // EvalRes our_eval;
          // our_eval.value = minneart_eval.value + velvety_eval.value;
          // our_eval.pdf
          //   = (minneart_imp*minneart_eval.pdf+velvety_imp*velvety_eval.pdf)
          //   / max(1e-20f,minneart_imp+velvety_imp);
          // return our_eval;
        }
        
        vec3h reflectance;
        vec3h horizonScatteringColor;
        half horizonScatteringFallOff;
        half backScattering;
        // Minneart minneart;
        // Velvety velvety;

        enum { bsdfType = Minneart::bsdfType | Velvety::bsdfType };
      };
      struct DD {
        inline __device__
        void make(HitBSDF &multi, bool dbg) const
        {
          multi.reflectance = reflectance;
          multi.horizonScatteringColor = horizonScatteringColor;
          multi.horizonScatteringFallOff = horizonScatteringFallOff;
          multi.backScattering = backScattering;
        }
        vec3f reflectance;
        vec3f horizonScatteringColor;
        float horizonScatteringFallOff;
        float backScattering;
      };
    };
    
  }
}
