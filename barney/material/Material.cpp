// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/Context.h"
#include "barney/material/AnariPBR.h"
#include "barney/material/AnariMatte.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"

namespace BARNEY_NS {
  namespace render {
    
    PossiblyMappedParameter::DD
    PossiblyMappedParameter::getDD(Device *device) 
    {
      PossiblyMappedParameter::DD dd;
      dd.type = type;
      switch(type) {
      case SAMPLER:
        dd.samplerID = sampler ? sampler->samplerID : -1;
        break;
      case ATTRIBUTE:
        dd.attribute = attribute;
        break;
      case VALUE:
        (vec4f&)dd.value = value;
        break;
      case INVALID:
        (vec4f&)dd.value = vec4f(0.f,0.f,0.f,0.f);
        break;
      }
      return dd;
    }
    
    void PossiblyMappedParameter::set(const vec3f  &v)
    {
      set(vec4f(v.x,v.y,v.z,1.f));
    }

    void PossiblyMappedParameter::set(const float &v)
    {
      set(vec4f(v,0.f,0.f,1.f));
    }

    void PossiblyMappedParameter::set(const vec4f &v)
    {
      type    = VALUE;
      sampler = {};
      value   = v;
    }

    void PossiblyMappedParameter::set(Sampler::SP s)
    {
      type = SAMPLER;
      sampler   = s;
    }

    void PossiblyMappedParameter::set(const std::string &attributeName)
    {
      sampler = {};
      type    = ATTRIBUTE;
      attribute = parseAttribute(attributeName);
    }
    
    HostMaterial::HostMaterial(SlotContext *slotContext)
      : barney_api::Material(slotContext->context),
        devices(slotContext->devices),
        materialRegistry(slotContext->materialRegistry),
        materialID(slotContext->materialRegistry->allocate())
    {
      assert(slotContext->context);
    }

    HostMaterial::~HostMaterial()
    {
      BN_TRACK_LEAKS(std::cout << "#barney: ~HostMaterial deconstructing"
                     << std::endl);
      materialRegistry->release(materialID);
    }
    
    HostMaterial::SP HostMaterial::create(SlotContext *slotContext,
                                          const std::string &type)
    {
#ifndef NDEBUG
      static std::set<std::string> alreadyCreated;
      if (alreadyCreated.find(type) == alreadyCreated.end()) {
        alreadyCreated.insert(type);
        if (Context::logging())
        std::cout << "#bn: creating (at least one of) material type '" << type << "'" << std::endl;
      }
#endif
      if (type == "AnariMatte" || type == "matte")
        return std::make_shared<AnariMatte>(slotContext); 
      if (type == "physicallyBased" || type == "AnariPBR")
        return std::make_shared<AnariPBR>(slotContext); 
      return std::make_shared<AnariPBR>(slotContext); 
    }

    void HostMaterial::commit()
    {
      for (auto device : *devices) {
        DeviceMaterial dd = getDD(device);
        materialRegistry->setMaterial(materialID,dd,device);
      }
      hasBeenCommittedAtLeastOnce = true;      
    }
  
  }
}
