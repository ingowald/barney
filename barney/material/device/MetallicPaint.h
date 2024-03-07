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

namespace barney {
  namespace render {

    // helper function which computes cosT^2 from cosI and eta
    inline __device__ float sqrCosT(const float cosI, const float eta)
    {
      return 1.0f - sqr(eta)*(1.0f - sqr(cosI));
    }

    //! \brief Computes fresnel coefficient for dielectric medium
    /*! \detailed Computes fresnel coefficient for media interface with
     *  relative refraction index eta. Eta is the outside refraction index
     *  divided by the inside refraction index. Both cosines have to be
     *  positive. */
    inline __device__ float fresnelDielectric(float cosI, float cosT, float eta)
    {
      const float Rper = (eta*cosI -     cosT) * rcp(eta*cosI +     cosT);
      const float Rpar = (    cosI - eta*cosT) * rcp(    cosI + eta*cosT);
      return 0.5f*(sqr(Rpar) + sqr(Rper));
    }

    /*! Computes fresnel coefficient for media interface with relative
     *  refraction index eta. Eta is the outside refraction index
     *  divided by the inside refraction index. The cosine has to be
     *  positive. */
    inline __device__ float fresnelDielectric(float cosI, float eta)
    {
      const float sqrCosT = render::sqrCosT(cosI, eta);
      if (sqrCosT < 0.0f) return 1.0f;
      return fresnelDielectric(cosI, sqrt(sqrCosT), eta);
    }

    inline __device__ float fresnelDielectricEx(float cosI, float &cosT, float eta)
    {
      const float sqrCosT = render::sqrCosT(cosI, eta);
      if (sqrCosT < 0.0f)
        {
          cosT = 0.0f;
          return 1.0f;
        }
      cosT = sqrt(sqrCosT);
      return fresnelDielectric(cosI, cosT, eta);
    }

    /*! dielectirclayer, but for metallicpaint, where transmittance, thickness, and wight are 1 */
    template<typename Substrate>
    struct DielectricLayer1 : public BSDF {
      inline __device__
      void init(float eta) { this->eta = eta; }

      inline __device__       
      vec3f getAlbedo(bool dbg=false) const
      { return vec3f(.5f); }
      
      inline __device__
      EvalRes eval(render::DG dg, vec3f wi, bool dbg=false) const
      { 
        // inline BSDF_EvalRes DielectricLayer_eval(const varying BSDF* uniform super,
        //                                          const vec3f& wo, const vec3f& wi)
        // {
        // const varying DielectricLayer* uniform self = (const varying DielectricLayer* uniform)super;

        // float cosThetaO = dot(wo, getN(super));
        const vec3f N = dg.N;
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

#if 0
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


    inline __device__ float roughnessToAlpha(float roughness)
    {
      // Roughness is squared for perceptual reasons
      return max(sqr(roughness), 0.001f);
    }

    // [Burley, 2012, "Physically Based Shading at Disney", Course Notes, v3]
    inline __device__ vec2f roughnessToAlpha(float roughness, float anisotropy)
    {
      float aspect = sqrt(1.f - 0.9f * anisotropy);
      return vec2f(max(sqr(roughness) / aspect, 0.001f),
                        max(sqr(roughness) * aspect, 0.001f));
    }


    
    struct GGXDistribution {
      inline __device__ void init(vec2f alpha) { this->alpha = to_half(alpha); }
      vec2h alpha;
    };
    struct GGXDistribution1 {
      inline __device__ void init(float alpha) { this->alpha = alpha; }
      inline __device__
      float evalLambda(const vec3f& wo) const
      {
        float cosThetaO = wo.z;
        float cosThetaO2 = sqr(cosThetaO);
        float invA2 = (sqr(wo.x * (float)alpha) + sqr(wo.y * (float)alpha)) / cosThetaO2;
        return 0.5f * (-1.f + sqrt(1.f+invA2));
      }
      inline __device__
      float evalVisible(const vec3f& wh, const vec3f& wo,
                                          float cosThetaOH, float& pdf) const
      {
        float cosThetaO = wo.z;
        float D = eval(wh);
        pdf = evalG1(wo, cosThetaOH) * abs(cosThetaOH) * D / abs(cosThetaO);
        return D;
      }
      inline __device__
      float evalG1(const vec3f& wo, float cosThetaOH) const
      {
        float cosThetaO = wo.z;
        if (cosThetaO * cosThetaOH <= 0.f)
          return 0.f;
        
        return rcp(1.f + evalLambda(wo));
      }
      inline __device__
      float evalG2(const vec3f& wo, const vec3f& wi, float cosThetaOH, float cosThetaIH) const
      {
        float cosThetaO = wo.z;
        float cosThetaI = wi.z;
        if (cosThetaO * cosThetaOH <= 0.f || cosThetaI * cosThetaIH <= 0.f)
          return 0.f;
        
        return rcp(1.f + evalLambda(wo) + evalLambda(wi));
      }
      inline __device__
      float eval(const vec3f& wh) const
      {
        float cosTheta = wh.z;
        float cosTheta2 = sqr(cosTheta);
        
        // float e = (sqr(wh.x / self.alpha.x) + sqr(wh.y / self.alpha.y)) / cosTheta2;
        float e = (sqr(wh.x / (float)alpha) + sqr(wh.y / (float)alpha)) / cosTheta2;
        // return rcp(pi * self.alpha.x * self.alpha.y * sqr(cosTheta2 * (1.f + e)));
        return rcp(pi * (float)alpha * (float)alpha * sqr(cosTheta2 * (1.f + e)));
      }
      
      half alpha;
    };
    
    struct FresnelSchlick1 {
      // inline __device__ FresnelSchlick1() {}
      inline __device__ void init(vec3f r, float g) { this->r = to_half(r); this->g = g; }
      
      inline __device__ vec3f eval(float cosI, bool dbg = false) const
      {
        const float c = 1.f - cosI;
        if (dbg)
          printf("c %f r %f %f %f g %f\n",
                 c,float(r.x),float(r.y),float(r.z), float(g));
        return lerp(sqr(sqr(c))*c, (vec3f)r, vec3f((float)g));
      }

      vec3h r; // reflectivity at normal incidence (0 deg)
      half  g;// reflectivity at grazing angle (90 deg)
    };

    template<typename Fresnel>
    struct MicrofacetConductor {
      enum { bsdfType = BSDF_GLOSSY_REFLECTION };
      inline __device__
      float importance() const { return luminance(fresnel.r); }
      
      inline __device__
      EvalRes eval(DG dg, vec3f wi, bool dbg = false) const
      {
        linear3f localFrame = owl::common::frame(dg.N);
        
        vec3f wo = dg.wo;
        float cosThetaO = dot(wo, dg.N);
        float cosThetaI = dot(wi, dg.N);
        if (cosThetaO <= 0.f || cosThetaI <= 0.f) {
          if (dbg) printf("no angle in microfacet\n");
          return EvalRes(vec3f(0.f),0.f);
        }
        
        EvalRes res;
        // Compute the microfacet normal
        vec3f wh = normalize(wi + wo);
        float cosThetaH = dot(wh, dg.N);
        float cosThetaOH = dot(wo, wh);
        float cosThetaIH = dot(wi, wh);
        
        // linear3f toLocal = transposed(getFrame(super));
        linear3f toLocal = localFrame.transposed();
        vec3f wo0 = toLocal * wo;
        vec3f wi0 = toLocal * wi;
        vec3f wh0 = toLocal * wh;
        
        vec3f F = fresnel.eval(cosThetaOH,dbg);
        float whPdf;
        float D = microfacet.evalVisible(wh0, wo0, cosThetaOH, whPdf);
        float G = microfacet.evalG2(wo0, wi0, cosThetaOH, cosThetaIH);

        // // Energy compensation
        // float Eo = MicrofacetAlbedoTable_eval(cosThetaO, roughness);
        // float Ei = MicrofacetAlbedoTable_eval(cosThetaI, roughness);
        // vec3f fms = self->fmsScale * ((1.f - Eo) * (1.f - Ei) * rcp(pi * (1.f - self->Eavg)) * cosThetaI);

  // float Eo = MicrofacetAlbedo_integrate(cosThetaO, roughness);
  // float Ei = MicrofacetAlbedo_integrate(cosThetaO, roughness);

        if (dbg) {
          printf("F %f %f %f D %f G %f\n",
                 F.x,F.y,F.z,D,G);
        }
        res.pdf = whPdf * rcp(4.f*abs(cosThetaOH));
        res.value = F * (D * G * rcp(4.f*cosThetaO));// + fms;
        
        return res;
      }
      
      
      inline __device__
      void init(Fresnel fresnel, float roughness)
      {
        // , float anisotropy --> 0.f
        // const float anisotropy = 0.f;
        
        // self->Eavg = MicrofacetAlbedoTable_evalAvg(roughness);
        // Eavg = MicrofacetAlbedoTable::evalAvg(roughness);
        // // vec3f Favg = fresnel->evalAvg(fresnel);
        // Favg = fresnel.evalAvg();
        // // self->fmsScale = sqr(Favg) * self->Eavg / (1.f - Favg * (1.f - self->Eavg)); // Stephen Hill's tweak
        // fmsScale = sqr(Favg)*Eavg / (1.f - Favg * (1.f-Eavg));
        
        // BSDF_Constructor(&self->super, Favg * self->Eavg, // TODO better estimate
        //                  BSDF_GLOSSY_REFLECTION,
        //                  MicrofacetConductor_eval, MicrofacetConductor_sample,
        //                  frame);
        // self->fresnel = fresnel;
        this->fresnel = fresnel;
        // self->microfacet = make_GGXDistribution(roughnessToAlpha(roughness, anisotropy));
        // microfacet.init(roughnessToAlpha(roughness,anisotropy));
        microfacet.init(roughnessToAlpha(roughness));
        // self->roughness = roughness;
        this->roughness = roughness;
      }
      
      Fresnel fresnel;
      GGXDistribution1 microfacet;
      half roughness;
      
      // Energy compensation [Kulla and Conty, 2017]
      half Eavg;
      vec3h fmsScale;
    };


    

    struct MetallicPaint {
      struct Substrate {
        inline __device__
        vec3f getAlbedo(bool dbg=false) const
        { return (vec3f)lambert.albedo; }
        
        inline __device__
        EvalRes eval(render::DG dg, vec3f wi, bool dbg=false) const
        {
          EvalRes lambert_eval = lambert.eval(dg,wi,dbg);
          float   lambert_imp  = lambert.importance();
          EvalRes facets_eval = facets.eval(dg,wi,dbg);
          float   facets_imp  = facets.importance();
          // EvalRes velvety_eval  = velvety.eval(dg,wi,dbg);
          // float   velvety_imp   = minneart.importance();
          EvalRes our_eval;
          our_eval.value
            = lambert_eval.value
            + facets_eval.value
            ;
          if (dbg) {
            printf("lambert %f %f %f\n",
                   lambert_eval.value.x,
                   lambert_eval.value.y,
                   lambert_eval.value.z);
            printf("facets %f %f %f\n",
                   facets_eval.value.x,
                   facets_eval.value.y,
                   facets_eval.value.z);
          }
          our_eval.pdf 
            = (lambert_imp*lambert_eval.pdf+facets_imp*facets_eval.pdf)
            / max(1e-20f,lambert_imp+facets_imp);
          return our_eval;
        }
        
        Lambert lambert;
        MicrofacetConductor<FresnelSchlick1> facets;
        enum { bsdfType = Lambert::bsdfType };
      };
      
      struct HitBSDF {
        inline __device__
        EvalRes eval(render::DG dg, vec3f wi, bool dbg=false) const
        { return dielectricLayer.eval(dg,wi,dbg); }
        
        inline __device__       
        vec3f getAlbedo(bool dbg=false) const
        { return dielectricLayer.getAlbedo(dbg); }
        
        DielectricLayer1<Substrate> dielectricLayer;
        enum { bsdfType = BSDF_SPECULAR_REFLECTION | Substrate::bsdfType };
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
          float flakeAmount = max(.1f,this->flakeAmount);
          vec3f r = flakeColor;
          float g = flakeAmount;
          FresnelSchlick1 fresnel; fresnel.init(r,g);
          multi.dielectricLayer.substrate.facets.init(fresnel,flakeSpread);
          multi.dielectricLayer.substrate.lambert.init(baseColor,dbg);
          multi.dielectricLayer.init(this->eta);
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
