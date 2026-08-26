// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/render/Ray.h"
#include "native/packedBSDF/PackedBSDF.h"
#include "native/render/HitAttributes.h"
#include "native/material/AnariMatte.h"
#include "native/material/AnariPBR.h"
#include "native/material/NVisii.h"

namespace BARNEY_NS {
  namespace native {
      
    struct DeviceMaterial {
      typedef enum {
        INVALID=0,
        TYPE_AnariMatte,
        TYPE_AnariPBR,
        TYPE_NVisii
      } Type;

#if RTC_DEVICE_CODE
      inline __rtc_device
      PackedBSDF createBSDF(const HitAttributes &hitData,
                            const Sampler::DD *samplers,
                            bool dbg=false) const;
      inline __rtc_device
      float getOpacity(const HitAttributes &hitData,
                       const Sampler::DD *samplers,
                       bool dbg) const;

      inline __rtc_device
      void setHit(Ray &ray,
                  const HitAttributes &hitData,
                  const Sampler::DD *samplers,
                  bool dbg=false) const;
#endif      
      Type type;
      union {
        AnariPBR::DD   anariPBR;
        AnariMatte::DD anariMatte;
        NVisii::DD     nvisii;
      };
    };

#if RTC_DEVICE_CODE
    inline __rtc_device
    PackedBSDF DeviceMaterial::createBSDF(const HitAttributes &hitData,
                                          const Sampler::DD *samplers,
                                          bool dbg) const
    {
      if (type == TYPE_AnariMatte)
        return anariMatte.createBSDF(hitData,samplers,dbg);
      if (type == TYPE_AnariPBR)
        return anariPBR.createBSDF(hitData,samplers,dbg);
      if (type == TYPE_NVisii)
        return nvisii.createBSDF(hitData,samplers,dbg);
#ifndef NDEBUG
      printf("#bn: DeviceMaterial::createBSDF encountered an invalid "
             "device material type (%i); most likely this is the app"
             " not having properly committed its material\n",(int)type);
#endif
      return packedBSDF::Invalid();
    }

    inline __rtc_device
    void DeviceMaterial::setHit(Ray &ray,
                                const HitAttributes &hitData,
                                const Sampler::DD *samplers,
                                bool dbg) const
    {
      ray.setHit(hitData.worldPosition,hitData.worldNormal,
                 hitData.t,createBSDF(hitData,samplers,dbg));
    }
#endif
  }
}
