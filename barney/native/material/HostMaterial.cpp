// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/Context.h"
#include "native/material/DeviceMaterial.h"
#include "native/material/AnariPBR.h"
#include "native/material/AnariMatte.h"
#include "native/material/NVisii.h"
#include "native/material/Glass.h"
#include "native/ModelSlot.h"
#include "native/Context.h"

namespace BARNEY_NS {
  namespace native {
    
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
    
    HostMaterial::HostMaterial(LDGContext *slotContext)
      : Object(slotContext->context),
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
    
    HostMaterial::SP HostMaterial::create(LDGContext *slotContext,
                                          const std::string &type)
    {
      if (type == "AnariMatte" || type == "matte")
        return std::make_shared<AnariMatte>(slotContext); 
      if (type == "physicallyBased" || type == "AnariPBR")
        return std::make_shared<AnariPBR>(slotContext); 
      if (type == "nvisii" || type == "NVisii")
        return std::make_shared<NVisii>(slotContext); 
      if (type == "glass" || type == "Glass")
        return std::make_shared<Glass>(slotContext); 
      return std::make_shared<AnariPBR>(slotContext); 
    }

    bool HostMaterial::setString(const std::string &member,
                                 const std::string &value)
    {
      if (Object::setString(member,value)) return true;
      if (member == "alphaMode") {
        if (value == "opaque")
          alphaMode = AlphaMode::Opaque;
        else if (value == "mask")
          alphaMode = AlphaMode::Mask;
        else if (value == "blend")
          alphaMode = AlphaMode::Blend;
        else
          warn_unsupported_member("string", "alphaMode");
        return true;
      }
      return false;
    }

    bool HostMaterial::set1f(const std::string &member, const float &value)
    {
      if (Object::set1f(member,value)) return true;
      if (member == "alphaCutoff") {
        alphaCutoff = value;
        return true;
      }
      return false;
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
