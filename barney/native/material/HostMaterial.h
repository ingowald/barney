// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/Object.h"
#include "native/render/HitAttributes.h"
#include "native/render/Sampler.h"
#include "native/render/MaterialRegistry.h"

namespace BARNEY_NS {
  namespace native {

    enum class AlphaMode : int { Opaque = 0, Mask, Blend };

    struct LDGContext;
    struct DeviceMaterial;
    
    struct PossiblyMappedParameter {
      typedef enum { INVALID=0, VALUE, ATTRIBUTE, SAMPLER } Type;

      PossiblyMappedParameter() = default;
      PossiblyMappedParameter(const vec3f v)
      { type = VALUE; value = vec4f(v.x,v.y,v.z,1.f); }
      PossiblyMappedParameter(float v)
      { type = VALUE; value = vec4f(v,0.f,0.f,1.f); }
      
      struct DD {
#if RTC_DEVICE_CODE
        inline __rtc_device
        vec4f eval(const HitAttributes &hitData,
                    const Sampler::DD *samplers,
                    bool dbg=false) const;
#endif
        Type type;
        union {
          rtc::float4          value;
          HitAttributes::Which attribute;
          int                  samplerID;
        };
      };

      void set(const float &v);
      void set(const vec3f &v);
      void set(const vec4f &v);
      void set(Sampler::SP sampler);
      void set(const std::string &attributeName);
      
      DD getDD(Device *device);
      
      Type type = VALUE;
      Sampler::SP          sampler;
      HitAttributes::Which attribute;
      vec4f               value { 0.f, 0.f, 0.f, 1.f };

      bool isConstantScalar(float v) const
      { return type == VALUE && value.x == v; }
      bool isConstantAlpha(float a) const
      { return type == VALUE && value.w == a; }
    };

    /*! barney 'virtual' material implementation that takes anari-like
      material paramters, and then builder device materials to be put
      into the device geometries */
    struct HostMaterial : public Object {
      typedef std::shared_ptr<HostMaterial> SP;

      /*! pretty-printer for printf-debugging */
      std::string toString() const override { return "<Material>"; }

      /*! device-data, as a union of _all_ possible device-side
        materials; we have to use a union here because no matter what
        virtual barney::Material gets created on the host, we have to
        have a single struct we put into the OWLGeom/SBT entry, else
        we'd have to have different OWLGeom type for different
        materials .... and possibly even change the actual OWLGeom
        (and even worse, its type) if the assigned material's type
        changes */

      HostMaterial(LDGContext *slotContext);
      virtual ~HostMaterial();

      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      void commit() override;
      bool setString(const std::string &member,
                     const std::string &value) override;
      bool set1f(const std::string &member, const float &value) override;
      bool set1i(const std::string &member, const int &value) override;
      /*! @} */
      // ------------------------------------------------------------------
      static HostMaterial::SP create(LDGContext *context,
                                     const std::string &type);
    
      virtual DeviceMaterial getDD(Device *device) = 0;

      void packCoverage(DeviceMaterial &dd, bool opacityIdenticallyOne) const;

      bool alphaModeSet = false;
      bool alphaCutoffSet = false;
      // Only consulted when alphaModeSet is true (see packCoverage).
      AlphaMode alphaMode = AlphaMode::Opaque;
      float alphaCutoff = 0.5f;

      /*! this material's index in the device list of all DeviceMaterials */
      const int materialID;

      bool hasBeenCommittedAtLeastOnce = false;
      DevGroup::SP const devices;
      
      // keep reference to material library, so it cannot die before
      // all materials are dead
      const MaterialRegistry::SP materialRegistry;
    };

#if RTC_DEVICE_CODE
    inline __rtc_device
    vec4f PossiblyMappedParameter::DD::eval(const HitAttributes &hitData,
                                            const Sampler::DD *samplers,
                                            bool dbg) const
    {
      if (type == VALUE) {
        return isnan(value.x) ? vec4f(0.f,0.f,0.f,1.f) : rtc::load(value);
      }
      if (type == ATTRIBUTE) {
        return hitData.get(attribute,dbg);
      } 
      if (type == SAMPLER) {
        return samplers[samplerID].eval(hitData,dbg);
      }
      return vec4f(0.f,0.f,0.f,1.f);
    }
#endif
    
  }
}
