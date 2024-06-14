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
#include "barney/render/packedBSDFs/principled/SheenDistribution.h"
#include "barney/render/packedBSDFs/principled/GGXDistribution.h"
#include "barney/render/packedBSDFs/principled/Fresnel.h"

namespace barney {
  namespace render {
    namespace packedBSDF {

      #define BSDF_DIFFUSE
      
      inline __device__ float MicrofacetDielectricReflectionAlbedoTable_eval
      (float cosTheta, float eta, float rouhness)
      { printf("missing\n"); return 0.f; };
      inline __device__ float MicrofacetSheenAlbedoTable_eval(float cosTheta, float rouhness)
      { printf("missing\n"); return 0.f; };
      inline __device__ float MicrofacetAlbedoTable_evalAvg(float roughness)
      { printf("missing\n"); return 0.f; };
      inline __device__ float fresnelDielectricAvg(float eta)
      { printf("missing\n"); return 0.f; };
      inline __device__ float MicrofacetDielectricReflectionAlbedoTable_evalAvg(float a, float b)
      { printf("missing\n"); return 0.f; };
      inline __device__ float MicrofacetAlbedoTable_eval(float a, float b)
      { printf("missing\n"); return 0.f; };
                           
      inline __device__ float sqrCosT(const float cosI, const float eta)
      {
        return 1.0f - sqr(eta) * (1.0f - sqr(cosI));
      }
      
      inline __device__ float refract(float cosI, float eta)
      {
        const float sqrcost = sqrCosT(cosI, eta);
        return sqrtf(max(sqrcost, 0.0f));
      }
      
      // struct TranslucentBase {
      // };

      // struct GlossyDiffuseBase {
      // };

      // struct MetalBase {
      // };
        
      // struct DielectricBase {
      //   TranslucentBase translucent;
      //   GlossyDiffuseBase glossyDiffuse;
      // };

      struct MetalBSDF {
        inline __device__ float importance() const { return weight; }
        inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg) const;
        float weight;
      };
      
      struct GlassBSDF {
        inline __device__ float importance() const { return weight; }
        inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg) const;
        float weight;
      };
      
      struct PlasticBSDF {
        inline __device__ float importance() const { return weight; }
        inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg) const;
        float weight;
      };

      inline __device__ EvalRes MetalBSDF::eval(DG dg, vec3f wi, bool dbg) const
      {
        return EvalRes::zero();
      }

      inline __device__ EvalRes GlassBSDF::eval(DG dg, vec3f wi, bool dbg) const
      {
        return EvalRes::zero();
      }
      
      inline __device__ EvalRes PlasticBSDF::eval(DG dg, vec3f wi, bool dbg) const
      {
        return EvalRes::zero();
      }
      
      struct BaseSubstrate {
        inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg) const;

        MetalBSDF   metal;
        GlassBSDF   glass;
        PlasticBSDF plastic;
      };

      /*! implement using only microfacetdielectriclayer, forcing any roughness to be non zero */
      struct CoatLayer
      {
        inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg) const;

        static inline __device__
        CoatLayer create(float weight, float ior, float eta,
                         vec3f color, float thickness, float roughness);
        
        // weight == 0.f means 'no coat layer'
        float weight;
        float ior;
        float eta;
        float3 coatColor;
        float thickness;
        float roughness;
        // Energy conservation/compensation [Kulla and Conty, 2017]
        float Ravg;
        float Eavg;
        float fmsScale;
        
        // MicrofacetDielectricLayer              
        BaseSubstrate substrateLayer;
      };

      template<typename FuzzBase>
      struct FuzzOver {
        inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg) const;

        float    weight; /* the 'sheen' parameter */
        float3   color;
        float    roughness;
        FuzzBase base;
      };
      
      struct PrincipledBSDF
      {
        inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg) const;

        float               opacity;
        FuzzOver<CoatLayer> fuzzLayer;
      };

      struct Principled
      {
        inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg) const;
      };
      

#ifdef __CUDACC__
        inline __device__
        CoatLayer CoatLayer::create(float weight, float ior, float eta,
                         vec3f color, float thickness, float roughness)
        {
          CoatLayer self;
          self.ior = ior < 1.f ? 1.f/ior : ior;
          self.roughness = max(roughness,0.05f);
          self.thickness = thickness;
          self.eta       = clamp(eta,1.f/3.f,3.f);
          self.coatColor = color;
          self.weight    = weight;

          self.Eavg = MicrofacetAlbedoTable_evalAvg(roughness);
          float Favg = fresnelDielectricAvg(eta);
          self.fmsScale = Favg * (1.f - self.Eavg) * rcp(1.f - Favg * self.Eavg);
          self.Ravg
            = (MicrofacetDielectricReflectionAlbedoTable_evalAvg(eta, roughness)
               + self.fmsScale * (1.f - self.Eavg))
            * weight; // add missing energy
          
          // XXX check
          // const vec3f albedo = substrate->albedo
          //     * ((1.0f - weight) + pow(transmittance, thickness) * (1.f - Favg));
          
          return self;
        }

      inline __device__ EvalRes BaseSubstrate::eval(DG dg, vec3f wi, bool dbg) const
      {
        EvalRes result = EvalRes::zero();
        float importanceSum = 0.f;
        { /* metal */
          EvalRes metal_res = metal.eval(dg,wi,dbg);
          result.value  += metal.weight       * metal_res.value;
          result.pdf    += metal.importance() * metal_res.pdf;
          importanceSum += metal.importance();
        }
        { /* glass */
          EvalRes glass_res = glass.eval(dg,wi,dbg);
          result.value  += glass.weight       * glass_res.value;
          result.pdf    += glass.importance() * glass_res.pdf;
          importanceSum += glass.importance();
        }
        { /* plastic */
          EvalRes plastic_res = plastic.eval(dg,wi,dbg);
          result.value  += plastic.weight       * plastic_res.value;
          result.pdf    += plastic.importance() * plastic_res.pdf;
          importanceSum += plastic.importance();
        }
        result.pdf /= importanceSum;
        return result;
      }


      
      inline __device__
      EvalRes CoatLayer::eval(DG dg, vec3f wi, bool dbg) const
      {
        vec3f wo = dg.wo;
        vec3f transmittance = vec3f(this->coatColor);
        linear3f frame = owl::common::frame(dg.Ns);
        
        float cosThetaO = dot(wo, dg.Ns);
        if (cosThetaO <= 0.f)
          return EvalRes::zero();
        
        float cosThetaI = dot(wi, dg.Ns);
        
        /* Evaluate the substrate */
        /* Ignore refraction */
        EvalRes substrate = substrateLayer.eval(dg, wi, dbg);
        
        /* Apply the coating medium absorption */                               
        /* Use refracted angles for computing the absorption path length */     
        float lengthO1 = rcp(refract(cosThetaO, eta)); /* rcp(cosThetaO1) */
        float lengthI1 = rcp(refract(abs(cosThetaI), eta)); /* rcp(cosThetaI1) */
        float length = lengthO1 + lengthI1;
        if (cosThetaI <= 0.f)                                     
          length *= 0.5f; /* for transmission, use the average length */
        substrate.value = lerp(this->weight,        
                               substrate.value,
                               substrate.value * pow(transmittance, thickness * length));
                                                                          
        /* Energy conservation */                                               
        float Eo = MicrofacetAlbedoTable_eval(cosThetaO, roughness);            
        float Ei = MicrofacetAlbedoTable_eval(fabsf(cosThetaI), roughness);       
        float Ro
          = (MicrofacetDielectricReflectionAlbedoTable_eval(cosThetaO, eta, roughness)
             + fmsScale * (1.f - Eo))
          * this->weight; /* add missing energy */
        float Ri
          = (MicrofacetDielectricReflectionAlbedoTable_eval(fabsf(cosThetaI),eta,roughness)                         + fmsScale * (1.f - Ei))                            
          * this->weight; /* add missing energy */                             
        float T;
#if 0
        if (scatteringType & ~BSDF_DIFFUSE)                                     
          T = min(1.f - Ro, 1.f - Ri); /* for generic (non-diffuse) substrates  
                                          [Kulla and Conty, 2017] */            
        else
#endif
          T = (1.f - Ro) * (1.f - Ri)                                           
            * rcp(1.f - Ravg); /* for diffuse substrates [Kelemen and    
                                       Szirmay-Kalos, 2001] */               
        substrate.value = substrate.value * T;                                  
                                                                          
        float coatPickProb = Ro;                                                
        float substratePickProb = 1.f - coatPickProb;                           
                                                                          
        if (cosThetaI > 0.f) {
          float anisotropy = 0.f;
          GGXDistribution microfacet
            = GGXDistribution::create(roughnessToAlpha(roughness, anisotropy));
          /* Compute the microfacet normal */                                   
          vec3f wh = normalize(wo + wi);                                        
          float cosThetaOH = dot(wo, wh);                                       
          
          /* Fresnel term */                                                    
          float F = fresnelDielectric(cosThetaOH, eta) * weight;      
                                                                          
          /* Evaluate the coating reflection */                                 
          float cosThetaIH = dot(wi, wh);                                       
                                                                          
          linear3f toLocal = frame.transposed();                 
          vec3f wo0 = toLocal * wo;                                             
          vec3f wi0 = toLocal * wi;                                             
          vec3f wh0 = toLocal * wh;                                             
                                                                          
          float whPdf;                                                          
          float D = microfacet.evalVisible(wh0, wo0, cosThetaOH, whPdf);  
          float G = microfacet.evalG2(wo0, wi0, cosThetaOH, cosThetaIH);  
                                                                          
          /* Energy compensation */                                             
          float fms = fmsScale                                             
            * ((1.f - Eo) * (1.f - Ei) * rcp((float)pi * (1.f - Eavg))   
               * cosThetaI);                                                 
                                                                          
          EvalRes coat;                                                    
          coat.pdf = whPdf * rcp(4.f * cosThetaOH);                             
          coat.value = vec3f(F * D * G * rcp(4.f * cosThetaO) + fms);      
                                                                          
          /* Compute the total reflection */                                    
          substrate.pdf =                                                       
            coatPickProb * coat.pdf + substratePickProb * substrate.pdf;      
          substrate.value = coat.value + substrate.value;                       
        } else {                                                                
          /* Return the substrate transmission */                               
          substrate.pdf *= substratePickProb;                                   
        }
        return substrate;
      }

        
      template<typename FuzzBase>
      inline __device__
      EvalRes FuzzOver<FuzzBase>::eval(DG dg, vec3f wi, bool dbg) const
      {
        vec3f wo = dg.wo;
        vec3f sheenColor = this->color;
        
        if (weight == 0.f) return base.eval(dg,wi,dbg);

        float cosThetaO = dot(wo, dg.Ns);
        if (cosThetaO <= 0.f)
          return EvalRes::zero();
        
        float cosThetaI = dot(wi, dg.Ns);

        /* Evaluate the substrate */
        EvalRes substrate = base.eval(dg,wi,dbg);

        /* Energy conservation */                              
        float Ro
          = MicrofacetSheenAlbedoTable_eval(cosThetaO, roughness) 
          * weight;                                           
        float Ri = MicrofacetSheenAlbedoTable_eval(fabsf(cosThetaI), roughness)
          * weight;                                           
        float T = min(1.f - Ro, 1.f - Ri);
        substrate.value = substrate.value * T;
        
        float coatPickProb = Ro;
        float substratePickProb = 1.f - coatPickProb;
        
        if (cosThetaI > 0.f) {
          SheenDistribution microfacet = SheenDistribution::create(roughnessToAlpha(roughness));
          
          /* Compute the microfacet normal */
          vec3f wh = normalize(wo + wi);
          float cosThetaH = dot(wh, dg.Ns);
          float cosThetaOH = dot(wo, wh);
          float cosThetaIH = dot(wi, wh);
          
          /* Evaluate the coating reflection */
          float D = microfacet.eval(cosThetaH);
          float G = microfacet.evalG2(cosThetaO, cosThetaI, cosThetaOH, cosThetaIH);
          
          EvalRes coat;                                                 
          coat.pdf = uniformSampleHemispherePDF();
          coat.value = sheenColor * (D * G * rcp(4.f * cosThetaO) * weight); 
                                                                        
          /* Compute the total reflection */                            
          substrate.pdf =                                               
            coatPickProb * coat.pdf + substratePickProb * substrate.pdf; 
          substrate.value = coat.value + substrate.value;               
        } else {                                                        
          /* Return the substrate transmission */                       
          substrate.pdf *= substratePickProb;                           
        }
        
        return substrate;
      }
      

      inline __device__ EvalRes PrincipledBSDF::eval(DG dg, vec3f wi, bool dbg) const
      {
        return fuzzLayer.eval(dg,wi,dbg);
      }


      inline __device__ EvalRes Principled::eval(DG dg, vec3f wi, bool dbg) const
      {
        return EvalRes::zero();
      }
      // struct Principled {
      //   inline Principled() = default;
      //   inline __device__ Principled(vec3f color, float avg_reflectance=.7f)
      //   {
      //     (vec3f&)this->albedo = avg_reflectance * color;
      //   }
      //   // inline __device__ vec3f getAlbedo(bool dbg) const;
      //   // inline __device__ float getOpacity(render::DG dg, bool dbg=false) const;
      //   inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg) const;
      //   inline __device__ void scatter(ScatterResult &scatter,
      //                                  const render::DG &dg,
      //                                  Random &random,
      //                                  bool dbg) const;
      //   float3 albedo;
      // };

      // inline __device__ EvalRes Principled::eval(DG dg, vec3f wi, bool dbg) const
      // {
      //   return EvalRes((const vec3f&)albedo,1.f);
      //   // return EvalRes::zero();
      // }

      // /*! simple omnidirectional phase function - scatter into any
      //     random direction */
      // inline __device__ void Principled::scatter(ScatterResult &scatter,
      //                                            const render::DG &dg,
      //                                            Random &random,
      //                                            bool dbg) const
      // {
      //   // see global illumination compendium, page 19
      //   float r1 = random();
      //   float r2 = random();
      //   // float phi = two_pi*r1;
      //   // float theta = acosf(1.f-2.f*r2);
      //   float x = cosf(two_pi*r1)*sqrtf(r2*(1.f-r2));
      //   float y = sinf(two_pi*r1)*sqrtf(r2*(1.f-r2));
      //   float z = (1.f-2.f*r2);
      //   float density = 1.f/(4.f*M_PI);
      //   scatter.pdf = density;
      //   scatter.f_r = (const vec3f&)albedo * density;
      //   scatter.dir = vec3f(x,y,z);
      // }
#endif
      
    }
  }
}

