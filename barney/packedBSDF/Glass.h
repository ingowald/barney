// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/render/DG.h"
#include "barney/render/HitAttributes.h"

namespace BARNEY_NS {
  namespace render {
    namespace fromOSPRAY {
      
      struct RefractionResult
      {
        vec3f outgoingDirection;
        float fresnelReflectionCoefficient;
        bool  mustReflect;
      };

      inline __rtc_device
      RefractionResult refractionDirection(const vec3f& incomingDir,
                                           const vec3f& surfaceNormal,
                                           const float eta,
                                           bool dbg = false)
      {
        RefractionResult res;
        res.mustReflect = false;

        float cos1 = dot(incomingDir, surfaceNormal);
        float cos1_abs = cos1;
        if (cos1_abs < 0.0) {
          cos1_abs = -cos1_abs;
        }
        // For head-on collisions the cosine of the angle of the ray with the surface
        // normal will be equal to one or minus one.
        bool headOn = (1.f - 1e-6f < cos1_abs && cos1_abs < 1.f - 1e-6f);
        if (headOn) {
          res.fresnelReflectionCoefficient = sqr((eta - 1.f)/(eta + 1.f));
          res.outgoingDirection = incomingDir;
        } else {
          vec3f dirParallelToNormal = cos1*surfaceNormal;
          vec3f dirPerpendicularToNormal = incomingDir - dirParallelToNormal;
          vec3f b = normalize(dirPerpendicularToNormal);
          float sin1 = dot(dirPerpendicularToNormal,b);
          float sin2 = sin1/eta;
          if (sin2 > 1.f) {// Total internal reflection.
            res.outgoingDirection = vec3f(0.0f);// Can't refract.
            res.fresnelReflectionCoefficient = 1.f;
            res.mustReflect = true;
          } else {// Refracting.
            float cos2 = sqrtf(clamp(1.f - sin2*sin2));
            vec3f N2 = normalize(dirParallelToNormal);
            res.outgoingDirection = N2*cos2 + b*sin2;
            float c1 = cos1_abs;
            float c2 = cos2;
            float r1 = (eta*c1 - c2)/(eta*c1 + c2);
            float r2 = (c1 - eta*c2)/(c1 + eta*c2);
            res.fresnelReflectionCoefficient = 0.5f*(r1*r1 + r2*r2);
          }
        }
        // Clamp in case of roundoff error.
        res.fresnelReflectionCoefficient = clamp(res.fresnelReflectionCoefficient);

        return res;
      }

      inline __rtc_device
      vec3f reflectionDirection(const vec3f &incomingDir,
                                const vec3f &normal)
      { // Imagine a particle traveling toward the surface and then bouncing off in the
        // reflection direction.
        
        // Parallel or anti-parallel direction.
        vec3f dirParallelToNormal = dot(incomingDir, normal) * normal;
        vec3f dirPerpendicularToNormal = incomingDir - dirParallelToNormal;
        
        // Outgoing direction.
        return (dirPerpendicularToNormal - dirParallelToNormal);
      }
      

      struct RobustDielectric
      {
        inline __rtc_device
        RobustDielectric(float eta) : eta(eta)
        {}
      
        inline __rtc_device
        EvalRes eval(render::DG dg, vec3f wi, bool dbg=false) const
        { return EvalRes::zero(); }

        inline __rtc_device
        SampleRes sample(const DG &dg,
                         Random &randomF,
                         bool dbg = false)
        {
          SampleRes res;
          res.pdf = BARNEY_INF;

          vec3f wo = dg.wo;
          vec3f incomingDir = neg(wo);// We want the actual ray direction into the surface.
          vec3f shadingNormal = dg.Ns;//self->dgeom.Ns;
          vec3f geometricNormal = dg.Ng;//self->dgeom.Ng;
          float eta = 1.f / this->eta;// ior(outside) / ior(inside)
          float glass_epsilon = 1.e-6f;
          bool canReflect = (eta < 1.f-glass_epsilon || eta > 1.f+glass_epsilon);
  
          bool be_careful = true;// Extra test for edge cases when the shading normal is different from the geometry normal.
          bool nudge = true;// When we have grazing angles. Only used if be_careful==true.
  
          if (be_careful) {
            // Flip the normals to be on the same side of the surface as the incoming ray.
            if (dot(incomingDir, shadingNormal) > 0) {
              shadingNormal = neg(shadingNormal);
            }
            if (dot(incomingDir, geometricNormal) > 0) {
              geometricNormal = neg(geometricNormal);
            }
          }
  
          float geometricCosine;
          vec3f dirR = reflectionDirection(incomingDir, shadingNormal);

          RefractionResult refractionRes = refractionDirection(incomingDir, shadingNormal, eta);
          vec3f dirT = refractionRes.outgoingDirection;
          float fresnelR = refractionRes.fresnelReflectionCoefficient;
          bool mustReflect = refractionRes.mustReflect;
  
          res.weight = vec3f(1.0f);

          // Sample the reflection or the transmission using the provided random number.
          if ((randomF() < fresnelR && canReflect) || mustReflect) {// Reflection
            res.wi = dirR;// Outgoing ray direction.
            res.type = SampleRes::SPECULAR_REFLECTION;
          
            if (be_careful) {
              geometricCosine = dot(res.wi, geometricNormal);
              if (geometricCosine >= 0) {
                // Reflecting as expected.
                if (nudge && (geometricCosine < glass_epsilon)) {// Grazing angle.
                  res.wi = res.wi + glass_epsilon*geometricNormal;// Nudge the ray away from the surface.
                  res.wi = normalize(res.wi);
                }
              } else {// geometricCosine < 0
                // We actually "refracted" since we are now on the other side of the surface.
                // Keep the ray type as reflection, but move the ray above the surface.
                res.wi = res.wi - 2.f*geometricNormal*geometricCosine;
              }
            }
          } else {// Transmission
            res.wi = dirT;// Outgoing ray direction.
            res.type = SampleRes::SPECULAR_TRANSMISSION;
            //res.weight = vec3f(rsqrtf(self->eta));// Solid angle compression.
    
            if (be_careful) {
              geometricCosine = dot(res.wi, geometricNormal);
              if (geometricCosine <= 0) {
                // Transmitting as expected.
                if (nudge && (geometricCosine > -glass_epsilon)) {// Grazing angle.
                  res.wi = res.wi - glass_epsilon*geometricNormal;// Nudge the ray away from the surface.
                  res.wi = normalize(res.wi);
                }
              } else {// geometricCosine > 0
                // We actually "reflected" since we are still on the same side of the surface.
                // Change ray type to reflection.
                res.type = SampleRes::SPECULAR_REFLECTION;
              }
            }
          }

          return res;
        }
      
        float eta;
      };
      
    } // :: packedBSDF::fromOSPRAY
    
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
        rtc::float3 attenuation;
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
        if (isShadowRay) 
          return 0.f;
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
        SampleRes sampleRes = fromOSPRAY::RobustDielectric(eta).sample(dg,random,dbg);
        scatter.f_r
          = sampleRes.weight;
        scatter.dir
          = sampleRes.wi;
        scatter.pdf
          = sampleRes.pdf;
        scatter.offsetDirection
          = (sampleRes.type == SampleRes::SPECULAR_TRANSMISSION)
          ? -1.f
          : +1.f;
        scatter.changedMedium
          = (sampleRes.type == SampleRes::SPECULAR_TRANSMISSION);
        scatter.type = ScatterResult::SPECULAR;
      }
      
      
      inline __rtc_device float Glass::pdf(DG dg, vec3f wi, bool dbg) const
      {
        return 0.f;
      }
      
    }
  }
}

