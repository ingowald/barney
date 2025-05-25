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

#include "barney/geometry/Geometry.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"

#include "barney/geometry/Triangles.h"
#include "barney/geometry/Spheres.h"
#include "barney/geometry/Cones.h"
#include "barney/geometry/Cylinders.h"
#include "barney/geometry/Capsules.h"

namespace BARNEY_NS {

  Geometry::PLD *Geometry::getPLD(Device *device)
  {
    assert(device);
    assert(device->contextRank() >= 0);
    assert(device->contextRank() < perLogical.size());
    return &perLogical[device->contextRank()];
  }

  Geometry::SP Geometry::create(Context *context,
                                DevGroup::SP devices,
                                const std::string &type)
  {
    if (type == "spheres")
      return std::make_shared<Spheres>(context,devices);
    if (type == "cones")
      return std::make_shared<Cones>(context,devices);
    if (type == "cylinders")
      return std::make_shared<Cylinders>(context,devices);
    if (type == "capsules")
      return std::make_shared<Capsules>(context,devices);
    if (type == "triangles")
      return std::make_shared<Triangles>(context,devices);
    
    context->warn_unsupported_object("Geometry",type);
    return {};
  }

  Geometry::Geometry(Context *context,
                     DevGroup::SP devices)
    : barney_api::Geometry(context),
      devices(devices)
  {
    perLogical.resize(devices->numLogical);
  }

  bool Geometry::set1i(const std::string &member,
                     const int   &value) 
  {
    if (member == "userID") {
      userID = value;
      return true; 
    } 
    
    return false;
  }
  
  Geometry::~Geometry()
  {
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      for (auto &geom : pld->triangleGeoms)
        if (geom) {
          // owlGeomRelease(geom);
          device->rtc->freeGeom(geom);
          geom = 0;
        }
      for (auto &geom : pld->userGeoms)
        if (geom) {
          // owlGeomRelease(geom);
          device->rtc->freeGeom(geom);
          geom = 0;
        }
    }
  }

  HostMaterial::SP Geometry::getMaterial() const
  {
    assert(this->material);
    return this->material;
  }
  
  void Geometry::setMaterial(HostMaterial::SP mat)
  {
    if (mat) {
      assert(mat->hasBeenCommittedAtLeastOnce);
    }
    this->material = mat;
  }
    
  void Geometry::writeDD(Geometry::DD &dd,
                         Device *device)
  {
    setAttributesOn(dd,device);
    dd.userID     = userID;
    dd.attributes = attributes.getDD(device);
    dd.materialID = getMaterial()->materialID;
  }  
  
  void Geometry::setAttributesOn(Geometry::DD &dd,
                                 Device *device)
  {
    dd.attributes = attributes.getDD(device);
    dd.materialID = material->materialID;  
  }
  
  bool Geometry::set1f(const std::string &member, const float &value)
  {
    return false;
  }
  
  bool Geometry::set3f(const std::string &member, const vec3f &value)
  {
    if (member == "attribute0") {
      attributes.attribute[0].constant = vec4f(value.x,value.y,value.z,1.f);
      return true;
    }
    if (member == "attribute1") {
      attributes.attribute[1].constant = vec4f(value.x,value.y,value.z,1.f);
      return true;
    }
    if (member == "attribute2") {
      attributes.attribute[2].constant = vec4f(value.x,value.y,value.z,1.f);
      return true;
    }
    if (member == "attribute3") {
      attributes.attribute[3].constant = vec4f(value.x,value.y,value.z,1.f);
      return true;
    }
    if (member == "color") {
      attributes.colorAttribute.constant = vec4f(value.x,value.y,value.z,1.f);
      return true;
    }
    
    return false;
  }
  
  bool Geometry::set4f(const std::string &member, const vec4f &value)
  {
    if (member == "attribute0") {
      attributes.attribute[0].constant = value;
      return true;
    }
    if (member == "attribute1") {
      attributes.attribute[1].constant = value;
      return true;
    }
    if (member == "attribute2") {
      attributes.attribute[2].constant = value;
      return true;
    }
    if (member == "attribute3") {
      attributes.attribute[3].constant = value;
      return true;
    }
    if (member == "color") {
      attributes.colorAttribute.constant = value;
      return true;
    }
    
    return false;
  }
  
  bool Geometry::setData(const std::string &member, const Data::SP &value)
  {
    if (member == "primitive.attribute0") {
      attributes.attribute[0].perPrim = value->as<PODData>();
      return true;
    }
    if (member == "primitive.attribute1") {
      attributes.attribute[1].perPrim = value->as<PODData>();
      return true;
    }
    if (member == "primitive.attribute2") {
      attributes.attribute[2].perPrim = value->as<PODData>();
      return true;
    }
    if (member == "primitive.attribute3") {
      attributes.attribute[3].perPrim = value->as<PODData>();
      return true;
    }
    if (member == "primitive.color") {
      attributes.colorAttribute.perPrim = value->as<PODData>();
      return true;
    }
    
    if (member == "vertex.attribute0") {
      attributes.attribute[0].perVertex = value->as<PODData>();
      return true;
    }
    if (member == "vertex.attribute1") {
      attributes.attribute[1].perVertex = value->as<PODData>();
      return true;
    }
    if (member == "vertex.attribute2") {
      attributes.attribute[2].perVertex = value->as<PODData>();
      return true;
    }
    if (member == "vertex.attribute3") {
      attributes.attribute[3].perVertex = value->as<PODData>();
      return true;
    }
    if (member == "vertex.color") {
      attributes.colorAttribute.perVertex = value->as<PODData>();
      return true;
    }
    
    return false;
  }
  
  bool Geometry::setObject(const std::string &member, const Object::SP &value)
  {
    if (member == "material") {
      HostMaterial::SP newMaterial = value->as<HostMaterial>();
      if (value && !newMaterial) {
        throw std::runtime_error("invalid material in geometry::set(\"material\"");
      }
      setMaterial(newMaterial);
      return true;
    }
    return false;
  }

}

