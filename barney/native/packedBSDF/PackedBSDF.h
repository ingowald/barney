// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/render/DG.h"
#include "native/packedBSDF/NVisii.h"
#include "native/packedBSDF/Glass.h"
#include "native/packedBSDF/Phase.h"
#include "native/packedBSDF/Lambertian.h"
#include "native/packedBSDF/PhysicallyBased.h"

namespace BARNEY_NS {
  namespace native {

    namespace packedBSDF {
      struct Invalid { };
    }

    struct PackedBSDF {
      typedef enum {
        INVALID=0, NONE=INVALID,
        /* henyey/greenstein*/TYPE_HGPhase,
        TYPE_SimplePhase,
        TYPE_Glass,
        TYPE_Lambertian,
        TYPE_NVisii,
        TYPE_PhysicallyBased
      } Type;
      struct Data {
        union {
          packedBSDF::SimplePhase simplePhase;
          packedBSDF::HGPhase     hgPhase;
          packedBSDF::Lambertian  lambertian;
          packedBSDF::Glass       glass;
          packedBSDF::NVisii      nvisii;
          packedBSDF::PhysicallyBased physicallyBased;
        };
      } data;

      Type type;

#if RTC_DEVICE_CODE
      inline __rtc_device PackedBSDF();
      inline __rtc_device PackedBSDF(Type type, Data data)
        : type(type), data(data) {}
      inline __rtc_device PackedBSDF(const packedBSDF::Invalid &invalid)
      { type = INVALID; }
      
      inline __rtc_device PackedBSDF(const packedBSDF::SimplePhase  &phase)
      { type = TYPE_SimplePhase; data.simplePhase = phase; }
      
      inline __rtc_device PackedBSDF(const packedBSDF::HGPhase  &phase)
      { type = TYPE_HGPhase; data.hgPhase = phase; }
      
      inline __rtc_device PackedBSDF(const packedBSDF::NVisii  &nvisii)
      { type = TYPE_NVisii; data.nvisii = nvisii; }
      inline __rtc_device PackedBSDF(const packedBSDF::Glass  &glass)
      { type = TYPE_Glass; data.glass = glass; }
      inline __rtc_device PackedBSDF(const packedBSDF::Lambertian  &lambertian)
      { type = TYPE_Lambertian; data.lambertian = lambertian; }
      inline __rtc_device PackedBSDF(const packedBSDF::PhysicallyBased &p)
      { type = TYPE_PhysicallyBased; data.physicallyBased = p; }
      
      inline __rtc_device
      EvalRes eval(DG dg, vec3f w_i, bool dbg=false) const;

      inline __rtc_device
      float pdf(DG dg, vec3f w_i, bool dbg=false) const;
      
      inline __rtc_device
      void scatter(ScatterResult &scatter,
                   const DG &dg,
                   Random &random,
                   bool dbg=false) const;

      /*! emitted radiance at this hit (already textured, since createBSDF
          bakes the evaluated emission sampler into the packed data);
          zero for non-emissive BSDF types. */
      inline __rtc_device
      vec3f getEmission(bool dbg=false) const;
      
#endif
    };

#if RTC_DEVICE_CODE
    inline __rtc_device
    EvalRes PackedBSDF::eval(DG dg, vec3f w_i, bool dbg) const
    {
      if (type == TYPE_SimplePhase)
        return data.simplePhase.eval(dg,w_i,dbg);
      
      if (type == TYPE_HGPhase)
        return data.hgPhase.eval(dg,w_i,dbg);
      
      if (type == TYPE_NVisii)
        return data.nvisii.eval(dg,w_i,dbg);
      if (type == TYPE_Glass)
        return data.glass.eval(dg,w_i,dbg);
      if (type == TYPE_Lambertian)
        return data.lambertian.eval(dg,w_i,dbg);
      if (type == TYPE_PhysicallyBased)
        return data.physicallyBased.eval(dg,w_i,dbg);
      return EvalRes();
    }
    
    inline __rtc_device
    float PackedBSDF::pdf(DG dg, vec3f w_i, bool dbg) const
    {
      if (type == TYPE_NVisii)
        return data.nvisii.pdf(dg,w_i,dbg);
      if (type == TYPE_Glass)
        return data.glass.pdf(dg,w_i,dbg);
      if (type == TYPE_Lambertian)
        return data.lambertian.pdf(dg,w_i,dbg);
      if (type == TYPE_SimplePhase)
        return data.simplePhase.pdf(dg,w_i,dbg);
      if (type == TYPE_HGPhase)
        return data.hgPhase.pdf(dg,w_i,dbg);
      if (type == TYPE_PhysicallyBased)
        return data.physicallyBased.pdf(dg,w_i,dbg);
      return 0.f;
    }
    
    inline __rtc_device
    void PackedBSDF::scatter(ScatterResult &scatter,
                             const DG &dg,
                             Random &random,
                             bool dbg) const
    {
      scatter.pdf = 0.f;
      if (type == TYPE_HGPhase)
        return data.hgPhase.scatter(scatter,dg,random,dbg);
      if (type == TYPE_SimplePhase)
        return data.simplePhase.scatter(scatter,dg,random,dbg);
      if (type == TYPE_NVisii)
        return data.nvisii.scatter(scatter,dg,random,dbg);
      if (type == TYPE_Glass)
        return data.glass.scatter(scatter,dg,random,dbg);
      if (type == TYPE_Lambertian)
        return data.lambertian.scatter(scatter,dg,random,dbg);
      if (type == TYPE_PhysicallyBased)
        return data.physicallyBased.scatter(scatter,dg,random,dbg);
    }

    inline __rtc_device
    vec3f PackedBSDF::getEmission(bool dbg) const
    {
      if (type == TYPE_PhysicallyBased)
        return (const vec3f &)data.physicallyBased.emissive;
      if (type == TYPE_HGPhase)
        return (const vec3f &)data.hgPhase.emission;
      return vec3f(0.f);
    }
#endif
  }
}
