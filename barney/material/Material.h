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

#include "barney/Object.h"
#include "barney/render/HitAttributes.h"
#include "barney/render/Sampler.h"
#include "barney/render/MaterialRegistry.h"

namespace BARNEY_NS {
  struct SlotContext;
  
  namespace render {

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
    };

    /*! barney 'virtual' material implementation that takes anari-like
      material paramters, and then builder barney::render:: style
      device materials to be put into the device geometries */
    struct HostMaterial : public barney_api::Material {
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

      HostMaterial(SlotContext *slotContext);
      virtual ~HostMaterial();

      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      void commit() override;
      /*! @} */
      // ------------------------------------------------------------------
      static HostMaterial::SP create(SlotContext *context,
                                     const std::string &type);
    
      virtual DeviceMaterial getDD(Device *device) = 0;

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
        if (samplerID < 0) return vec4f(0.f,0.f,0.f,1.f);
        if (samplerID != 0) {
          printf("SAMPLER %i\n",samplerID);
          return vec4f(0.f);
        }
        return samplers[samplerID].eval(hitData,dbg);
      }
      return vec4f(0.f,0.f,0.f,1.f);
    }
#endif
    
  }
}
