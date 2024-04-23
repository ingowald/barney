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

namespace barney {
  namespace render {
    namespace packedBSDF {
      
      struct VisRTX {
        inline __device__ vec3f getAlbedo(bool dbg) const;
        inline __device__ float getOpacity(render::DG dg, bool dbg=false) const;
        inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg) const;

        static inline __device__ VisRTX make_matte(const vec3f v);
        
        vec3h baseColor;
        half  opacity;
        half  metallic;
        half  roughness;
        half  ior;
      };

      inline __device__ float pow2(float f) { return f*f; }
      inline __device__ float pow5(float f) { return pow2(pow2(f))*f; }
      inline __device__ float mix(float f, float a, float b) { return (1.f-f)*a + f*b; }
      inline __device__ vec3f mix(vec3f f, vec3f a, vec3f b)
      { return vec3f(mix(f.x,a.x,b.x),mix(f.y,a.y,b.y),mix(f.z,a.z,b.z)); }
      inline __device__ float heaviside(float f) { return (f<0.f)?0.f:1.f; }

    
      inline __device__ vec3f VisRTX::getAlbedo(bool dbg) const
      { return baseColor; }
    
      inline __device__ EvalRes VisRTX::eval(DG dg, vec3f wi, bool dbg) const
      {
        // -----------------------------------------------------------------------------
        // 'unpack' compressed params
        // -----------------------------------------------------------------------------
        const float roughness = this->roughness;
        const float opacity   = this->opacity;
        const float metallic  = this->metallic;
        const vec3f baseColor = this->baseColor;
        const float ior       = this->ior;
        vec3f lightDir = wi;
        vec3f viewDir  = dg.wo;
        vec3f hit_Ns = dg.Ns;
        /* visrtx evaluates for a specific light, we compute BRDF - so
           light intensity gets multiplied in later on... just set to
           1 */
        vec3f lightIntensity = vec3f(1.f);
      
        // -----------------------------------------------------------------------------
        // actual shading/evaluation code
        // -----------------------------------------------------------------------------
        const vec3f H = normalize(lightDir + viewDir);
        const float NdotH = dot(hit_Ns, H);
        const float NdotL = dot(hit_Ns, lightDir);
        const float NdotV = dot(hit_Ns, viewDir);
        const float VdotH = dot(viewDir, H);
        const float LdotH = dot(lightDir, H);
      
        // Alpha
        const float alpha = pow2(roughness) * opacity;
      
        // Fresnel
        const vec3f f0 =
          mix(vec3f(pow2((1.f - ior) / (1.f + ior))),
              baseColor,
              metallic);
        const vec3f F = f0 + (vec3f(1.f) - f0) * pow5(1.f - fabsf(VdotH));

        // Metallic materials don't reflect diffusely:
        const vec3f diffuseColor =
          mix(baseColor, vec3f(0.f), metallic);

        const vec3f diffuseBRDF =
          (vec3f(1.f) - F) * float(M_1_PI) * diffuseColor * fmaxf(0.f, NdotL);

        // GGX microfacet distribution
        const float D = (alpha * alpha * heaviside(NdotH))
          / (float(M_PI) * pow2(NdotH * NdotH * (alpha * alpha - 1.f) + 1.f));

        // Masking-shadowing term
        const float G =
          ((2.f * fabsf(NdotL) * heaviside(LdotH))
           / (fabsf(NdotL)
              + sqrtf(alpha * alpha + (1.f - alpha * alpha) * NdotL * NdotL)))
          * ((2.f * fabsf(NdotV) * heaviside(VdotH))
             / (fabsf(NdotV)
                + sqrtf(alpha * alpha + (1.f - alpha * alpha) * NdotV * NdotV)));

        const float denom = 4.f * fabsf(NdotV) * fabsf(NdotL);
        const vec3f specularBRDF = denom != 0.f ? (F * D * G) / denom : vec3f(0.f);

        return {(diffuseBRDF + specularBRDF) * lightIntensity, opacity};
      }
      
      inline __device__ VisRTX VisRTX::make_matte(const vec3f albedo)
      {
        VisRTX v;
        v.metallic  = 0.f;
        v.roughness = 0.f;
        v.opacity   = 1.f;
        v.ior       = 1.f;
        v.baseColor = albedo;
        return v;
      }
      
    }    
  }
}

