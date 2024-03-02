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

namespace barney {
  namespace render {

    typedef uint32_t BSDFType;

#define BSDF_SPECULAR_REFLECTION   (1<<0)  /*!< perfect specular light reflection   */
#define BSDF_GLOSSY_REFLECTION     (1<<1)  /*!< glossy light reflection             */
#define BSDF_DIFFUSE_REFLECTION    (1<<2)  /*!< diffuse light reflection            */
#define BSDF_SPECULAR_TRANSMISSION (1<<3)  /*!< perfect specular light transmission */
#define BSDF_GLOSSY_TRANSMISSION   (1<<4)  /*!< glossy light transmission           */
#define BSDF_DIFFUSE_TRANSMISSION  (1<<5)  /*!< diffuse light transmission          */

/*! diffuse reflections and transmissions */
#define BSDF_DIFFUSE      (BSDF_DIFFUSE_REFLECTION   | BSDF_DIFFUSE_TRANSMISSION)

/*! glossy reflections and transmissions */
#define BSDF_GLOSSY       (BSDF_GLOSSY_REFLECTION    | BSDF_GLOSSY_TRANSMISSION)

/*! perfect specular reflections and transmissions */
#define BSDF_SPECULAR     (BSDF_SPECULAR_REFLECTION  | BSDF_SPECULAR_TRANSMISSION)

/*! all possible reflections */
#define BSDF_REFLECTION   (BSDF_DIFFUSE_REFLECTION   | BSDF_GLOSSY_REFLECTION   | BSDF_SPECULAR_REFLECTION)

/*! all possible transmissions */
#define BSDF_TRANSMISSION (BSDF_DIFFUSE_TRANSMISSION | BSDF_GLOSSY_TRANSMISSION | BSDF_SPECULAR_TRANSMISSION)

/*! all non-dirac types */
#define BSDF_SMOOTH       (BSDF_DIFFUSE | BSDF_GLOSSY)

/*! no component set */
#define BSDF_NONE         0

/*! all components set */
#define BSDF_ALL          (BSDF_REFLECTION | BSDF_TRANSMISSION)

    inline __device__
    float luminance(vec3f c)
    { return 0.212671f*c.x + 0.715160f*c.y + 0.072169f*c.z; }

    struct BSDF {
      inline __device__ float importance() const { return luminance(albedo); }
      inline __device__ void init(vec3f albedo, bool dbg=false)
      {
        this->albedo = albedo; //this->importance = importance;
      }
      vec3h albedo;
      // half  importance;
    };
    struct Minneart : public BSDF {
      inline __device__ void init(vec3f R, float b, bool dbg = false) {
        BSDF::init(R); this->b = b;
        if (dbg) printf("Minneart %f %f %f f %f\n",
                        float(albedo.x),
                        float(albedo.y),
                        float(albedo.z),
                        float(this->b));
      }
      inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg = false) const
      {
        EvalRes res;
        const float cosThetaI = clamp(dot(wi,dg.N));
        const float backScatter = powf(clamp(dot(dg.wo,wi)), (float)b);
        res.pdf = cosineSampleHemispherePDF(cosThetaI);
        res.value = (backScatter * cosThetaI * one_over_pi) * (vec3f)albedo;
        return res;
      }
      
      enum { bsdfType = BSDF_DIFFUSE_REFLECTION };
      /*! The amount of backscattering. A value of 0 means lambertian
       *  diffuse, and inf means maximum backscattering. */
      half b;
    };
    struct Velvety : public BSDF {
      inline __device__ void init(vec3f R, float f, bool dbg = false)
      { BSDF::init(R); this->f = f;
        if (dbg) printf("Velvety %f %f %f f %f\n",
                        float(albedo.x),
                        float(albedo.y),
                        float(albedo.z),
                        float(this->f));
      }
      inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg = false) const
      {
        EvalRes res;
        const float cosThetaO = clamp(dot(dg.wo,dg.N));
        const float cosThetaI = clamp(dot(wi,dg.N));
        const float sinThetaO = sqrt(1.0f - cosThetaO * cosThetaO);
        const float horizonScatter = pow(sinThetaO, (float)f);
        res.pdf = cosineSampleHemispherePDF(cosThetaI);
        res.value =  (horizonScatter * cosThetaI * one_over_pi) * (vec3f)albedo;
        return res;
      }
      enum { bsdfType = BSDF_DIFFUSE_REFLECTION };
      /*! The falloff of horizon scattering. 0 no falloff,
       *  and inf means maximum falloff. */
      half f;
    };
    struct Velvet {
      struct HitBSDF {
        inline __device__
        vec3f getAlbedo(bool dbg=false) const { return (vec3f)minneart.albedo+(vec3f)velvety.albedo; }
        // inline __device__ vec3f albedo() const { return minneart.albedo + velvety.albedo; }
        // inline __device__ int type() const { return minneart.type() | velvety.type(); }
        
        inline __device__
        EvalRes eval(render::DG dg, vec3f wi, bool dbg=false) const
        {
          if (dbg)
            printf("----- Velvet wo %f %f %f N %f %f %f wi %f %f %f\n",
                   dg.wo.x,
                   dg.wo.y,
                   dg.wo.z,
                   dg.N.x,
                   dg.N.y,
                   dg.N.z,
                   wi.x,
                   wi.y,
                   wi.z);
          EvalRes minneart_eval = minneart.eval(dg,wi,dbg);
          float   minneart_imp  = minneart.importance();
          EvalRes velvety_eval  = velvety.eval(dg,wi,dbg);
          float   velvety_imp   = minneart.importance();
          EvalRes our_eval;
          our_eval.value = minneart_eval.value + velvety_eval.value;
          our_eval.pdf
            = (minneart_imp*minneart_eval.pdf+velvety_imp*velvety_eval.pdf)
            / max(1e-20f,minneart_imp+velvety_imp);
          if (dbg) printf(" --> value %f %f %f pdf %f\n",
                          our_eval.value.x,
                          our_eval.value.y,
                          our_eval.value.z,
                          our_eval.pdf);
          return our_eval;
        }
        
        Minneart minneart;
        Velvety velvety;

        enum { bsdfType = Minneart::bsdfType | Velvety::bsdfType };
      };
      struct DD {
        inline __device__
        void make(HitBSDF &multi, bool dbg) const
        {
          multi.minneart.init(reflectance,backScattering,dbg);
          multi.velvety.init(horizonScatteringColor,horizonScatteringFallOff,dbg);
        }
        vec3f reflectance;
        vec3f horizonScatteringColor;
        float horizonScatteringFallOff;
        float backScattering;
      };
    };
    
  }
}
