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

#include "barney/material/AnariPBR.h"
#include "barney/material/AnariMatte.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"

namespace barney {
  namespace render {
    
    void PossiblyMappedParameter::make(DD &dd, int deviceID) const
    {
      dd.type = type;
      switch(type) {
      case SAMPLER:
        dd.samplerID = sampler ? sampler->samplerID : -1;
        break;
      case ATTRIBUTE:
        dd.attribute = attribute;
        break;
      case VALUE:
        dd.value = value;
        break;
      case INVALID:
        dd.value = make_float4(0.f,0.f,0.f,0.f);
        break;
      }
    }
    
    void PossiblyMappedParameter::set(const vec3f  &v)
    {
      set(make_float4(v.x,v.y,v.z,1.f));
    }

    void PossiblyMappedParameter::set(const float &v)
    {
      set(make_float4(v,0.f,0.f,1.f));
    }

    void PossiblyMappedParameter::set(const vec4f  &v)
    {
      set(make_float4(v.x,v.y,v.z,v.w));
    }

    void PossiblyMappedParameter::set(const float4 &v)
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
    
    HostMaterial::HostMaterial(Context *context, int slot)
      : SlottedObject(context,slot),
        materialRegistry(context->getSlot(slot)->materialRegistry),
        materialID(context->getSlot(slot)->materialRegistry->allocate())
    {}

    HostMaterial::~HostMaterial()
    {
      materialRegistry->release(materialID);
    }
    
    void HostMaterial::setDeviceDataOn(OWLGeom geom) const
    {
      owlGeomSet1i(geom,"materialID",materialID);
    }

    HostMaterial::SP HostMaterial::create(Context *context,
                                          int slot,
                                          const std::string &type)
    {
#ifndef NDEBUG
      static std::set<std::string> alreadyCreated;
      if (alreadyCreated.find(type) == alreadyCreated.end()) {
        alreadyCreated.insert(type);
        std::cout << "#bn: creating (at least one of) material type '" << type << "'" << std::endl;
      }
#endif
      // if (type == "matte")
      //   return std::make_shared<AnariMatte>(owner);
      // ==================================================================
      // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
      // specifically for anari layer:
      if (type == "AnariMatte")
        return std::make_shared<AnariMatte>(context,slot); 
      if (type == "physicallyBased" || type == "AnariPBR")
        return std::make_shared<AnariPBR>(context,slot); 
      // specifically for anari layer:
      // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      // ==================================================================
      return std::make_shared<AnariPBR>(context,slot); 
      // if (type == "velvet")
      //   return std::make_shared<VelvetMaterial>(dg);
      // if (type == "blender")
      //   return std::make_shared<BlenderMaterial1>(dg); 
      // if (type == "glass")
      //   return std::make_shared<GlassMaterial>(dg); 
      // if (type == "metal")
      //   return std::make_shared<MetalMaterial>(dg); 
      // if (type == "plastic")
      //   return std::make_shared<PlasticMaterial>(dg);
      // if (type == "metallic_paint")
      //   return std::make_shared<MetallicPaintMaterial>(dg);
      // if (type == "velvet")
      //   return std::make_shared<VelvetMaterial>(dg);
      // else
      // if (type == "physicallyBased")
      //   return std::make_shared<AnariPhysicalMaterial>(dg);
      // iw - "eventually" we should have different materials like
      // 'matte' and 'glass', 'metal' etc here, but for now, let's just
      // ignore the type and create a single one thta contains all
      // fields....
      // return std::make_shared<MiniMaterial>(dg);
    }

    void HostMaterial::commit()
    {
      SlottedObject::commit();
      DeviceMaterial dd;
      for (int devID=0;devID<getDevGroup()->size();devID++) {
        this->createDD(dd,devID);
        materialRegistry->setMaterial(materialID,dd,devID);
      }
      hasBeenCommittedAtLeastOnce = true;      
    }
  
  }
}
