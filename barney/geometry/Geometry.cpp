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

namespace barney {

  Geometry::SP Geometry::create(ModelSlot *owner,
                                const std::string &type)
  {
    if (type == "spheres")
      return std::make_shared<Spheres>(owner);
#if 0
    if (type == "cones")
      return std::make_shared<Cones>(owner);
#endif
    if (type == "cylinders")
      return std::make_shared<Cylinders>(owner);
    if (type == "capsules")
      return std::make_shared<Capsules>(owner);
    if (type == "triangles")
      return std::make_shared<Triangles>(owner);
    
    owner->context->warn_unsupported_object("Geometry",type);
    return {};
  }

  Geometry::Geometry(ModelSlot *owner)
    : SlottedObject(owner),
      material(owner->getDefaultMaterial())
  {}

  Geometry::~Geometry()
  {
    for (auto &geom : triangleGeoms)
      if (geom) { owlGeomRelease(geom); geom = 0; }
    for (auto &geom : userGeoms)
      if (geom) { owlGeomRelease(geom); geom = 0; }
    for (auto &group : secondPassGroups)
      if (group) { owlGroupRelease(group); group = 0; }
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
    
  
  
  OWLContext Geometry::getOWL() const
  { return owner->getOWL(); }

  void Geometry::addVars(std::vector<OWLVarDecl> &vars, int base)
  {
    vars.push_back({"materialID",OWL_INT,OWL_OFFSETOF(DD,materialID)});
    vars.push_back({"attributes",OWL_USER_TYPE(GeometryAttributes::DD),
        OWL_OFFSETOF(DD,attributes)});
  }


  void Geometry::setAttributesOn(OWLGeom geom)
  {
    auto set = [&](GeometryAttribute::DD &out, const GeometryAttribute &in,
                   const int devID,
                   const std::string &dbgName)
    {
      if (in.perVertex) {
        out.scope = GeometryAttribute::PER_VERTEX;
        out.fromArray.type = in.perVertex->type;
        out.fromArray.ptr  = owlBufferGetPointer(in.perVertex->owl,devID);
        out.fromArray.size = in.perVertex->count;
      } else if (in.perPrim) {
        out.scope = GeometryAttribute::PER_PRIM;
        out.fromArray.type = in.perPrim->type;
        out.fromArray.ptr  = owlBufferGetPointer(in.perPrim->owl,devID);
        out.fromArray.size = in.perPrim->count;
      } else {
        out.scope = GeometryAttribute::CONSTANT;
        (vec4f&)out.value = in.constant;
      }
      // PRINT(out.scope);
    };
    
    GeometryAttributes::DD dd;
    for (int devID=0;devID<owner->devGroup->size();devID++) {
      for (int i=0;i<attributes.count;i++) {
        const auto &in = attributes.attribute[i];
        auto &out = dd.attribute[i];
        set(out,in,devID,"attr"+std::to_string(i));
      }
      set(dd.colorAttribute,attributes.colorAttribute,devID,"color");
      owlGeomSetRaw(geom,"attributes",&dd,devID);
    }
  }
  

  bool Geometry::set1f(const std::string &member, const float &value)
  {
    return false;
  }
  
  bool Geometry::set3f(const std::string &member, const vec3f &value)
  {
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

