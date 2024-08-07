/*
 * Copyright (c) 2019++ NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "barney/material/device/DG.h"
#include "barney/material/device/BSDF.h"
#include "barney/material/bsdfs/MultiBSDF.h"

namespace barney {
  namespace render {

    struct VisRTX {
      struct DD;
      struct HitBSDF;
        
      struct HitBSDF {
        inline __device__ vec3f getAlbedo(bool dbg=false) const;
        inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg=false) const;
        inline __device__ DD unpack() const;
        
        vec3h baseColor;
        half  opacity;
        half  metallic;
        half  roughness;
        half  ior;
      };
      struct DD {
        inline __device__ void make(HitBSDF &multi, vec3f geometryColor, bool dbg) const;
        
        vec3f baseColor;
        float opacity;
        float metallic;
        float roughness;
        float ior;
      };
    };

    
    inline __device__ float pow2(float f) { return f*f; }
    inline __device__ float pow5(float f) { return pow2(pow2(f))*f; }
    inline __device__ float mix(float f, float a, float b) { return (1.f-f)*a + f*b; }
    inline __device__ vec3f mix(vec3f f, vec3f a, vec3f b)
    { return vec3f(mix(f.x,a.x,b.x),mix(f.y,a.y,b.y),mix(f.z,a.z,b.z)); }
    inline __device__ float heaviside(float f) { return (f<0.f)?0.f:1.f; }

    
    inline __device__ vec3f VisRTX::HitBSDF::getAlbedo(bool dbg) const
    { return baseColor; }
    
    inline __device__ EvalRes VisRTX::HitBSDF::eval(DG dg, vec3f wi, bool dbg) const
    {
      DD matValues = this->unpack();
      vec3f lightDir = wi;
      vec3f viewDir  = dg.wo;
      vec3f hit_Ns = dg.Ns;
      /* visrtx evaluates for a specific light, we compute BRDF - so
         light intensity gets multiplied in later on... just set to
         1 */
      vec3f lightIntensity = vec3f(1.f);
      
      const vec3f H = normalize(lightDir + viewDir);
      const float NdotH = dot(hit_Ns, H);
      const float NdotL = dot(hit_Ns, lightDir);
      const float NdotV = dot(hit_Ns, viewDir);
      const float VdotH = dot(viewDir, H);
      const float LdotH = dot(lightDir, H);
      
      // Alpha
      const float alpha = pow2(matValues.roughness) * matValues.opacity;
      
      // Fresnel
      const vec3f f0 =
        mix(vec3f(pow2((1.f - matValues.ior) / (1.f + matValues.ior))),
            matValues.baseColor,
            matValues.metallic);
      const vec3f F = f0 + (vec3f(1.f) - f0) * pow5(1.f - fabsf(VdotH));

      // Metallic materials don't reflect diffusely:
      const vec3f diffuseColor =
        mix(matValues.baseColor, vec3f(0.f), matValues.metallic);

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

      return {(diffuseBRDF + specularBRDF) * lightIntensity, matValues.opacity};
    }
    
    inline __device__ void VisRTX::DD::make(VisRTX::HitBSDF &multi,
                                            vec3f geometryColor,
                                            bool dbg) const
    {
      multi.baseColor = isnan(geometryColor.x) ? baseColor : geometryColor;
      multi.opacity   = opacity;
      multi.metallic  = metallic;
      multi.roughness = roughness;
      multi.ior       = ior;
    }
    
    inline __device__ VisRTX::DD VisRTX::HitBSDF::unpack() const
    {
      VisRTX::DD matValues;
      matValues.baseColor = baseColor;
      matValues.opacity   = opacity;
      matValues.metallic  = metallic;
      matValues.roughness = roughness;
      matValues.ior       = ior;
      return matValues;
    }
  }
}
