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

  
  OWLContext Geometry::getOWL() const
  { return owner->getOWL(); }

  void Geometry::addVars(std::vector<OWLVarDecl> &vars, int base)
  {
    // Material::addVars(vars,base+OWL_OFFSETOF(DD,material));
    vars.push_back({"materialID",OWL_INT,OWL_OFFSETOF(DD,materialID)});
    vars.push_back({"attributes",OWL_USER_TYPE(GeometryAttributes),
        OWL_OFFSETOF(DD,attributes)});
  }


  void Geometry::setAttributesOn(OWLGeom geom)
  {
    GeometryAttributes::DD dd;
    for (int devID=0;devID<owner->devGroup->size();devID++) {
      for (int i=0;i<attributes.count;i++) {
        const auto &in = attributes.attribute[i];
        auto &out = dd.attribute[i];
        if (in.perVertex) {
          out.scope = GeometryAttribute::PER_VERTEX;
          out.fromArray.type = in.perVertex->type;
          out.fromArray.ptr  = owlBufferGetPointer(in.perVertex->owl,devID);
        } else if (in.perPrim) {
          out.scope = GeometryAttribute::PER_PRIM;
          out.fromArray.type = in.perPrim->type;
          out.fromArray.ptr  = owlBufferGetPointer(in.perPrim->owl,devID);
        } else {
          out.scope = GeometryAttribute::CONSTANT;
          (vec4f&)out.value = in.constant;
        }
      }
      owlGeomSetRaw(geom,"attributes",&dd,devID);
    }
  }
  

  bool Geometry::set1f(const std::string &member, const float &value)
  {
    // if (member == "material.transmission") {
    //   material->set1f("transmission",value);
    //   return true;
    // }
    // if (member == "material.ior") {
    //   material->set1f("ior",value);
    //   return true;
    // }
    // if (member == "material.metallic") {
    //   material->set1f("metallic",value);
    //   return true;
    // }
    return false;
  }
  
  bool Geometry::set3f(const std::string &member, const vec3f &value)
  {
    // if (member == "material.baseColor") {
    //   material->set3f("baseColor",value);
    //   return true;
    // }
    return false;
  }
  
  bool Geometry::setData(const std::string &member, const Data::SP &value)
  {
    auto *attribute = attributes.attribute;
    if (member == "primitive.attribute0") {
      attribute[0].perPrim = value->as<PODData>();
      return true;
    }
    if (member == "primitive.attribute1") {
      attribute[1].perPrim = value->as<PODData>();
      return true;
    }
    if (member == "primitive.attribute2") {
      attribute[2].perPrim = value->as<PODData>();
      return true;
    }
    if (member == "primitive.attribute3") {
      attribute[3].perPrim = value->as<PODData>();
      return true;
    }
    
    if (member == "vertex.attribute0") {
      attribute[0].perVertex = value->as<PODData>();
      return true;
    }
    if (member == "vertex.attribute1") {
      attribute[1].perVertex = value->as<PODData>();
      return true;
    }
    if (member == "vertex.attribute2") {
      attribute[2].perVertex = value->as<PODData>();
      return true;
    }
    if (member == "vertex.attribute3") {
      attribute[3].perVertex = value->as<PODData>();
      return true;
    }
    return false;
  }
  
  bool Geometry::setObject(const std::string &member, const Object::SP &value)
  {
    if (member == "material") {
      // material->setObject("colorTexture",value);
      material = value->as<HostMaterial>();
      if (!material)
        throw std::runtime_error("invalid material in geometry::set(\"material\"");
      return true;
    }
    // if (member == "material.colorTexture") {
    //   material->setObject("colorTexture",value);
    //   return true;
    // }
    // if (member == "material.alphaTexture") {
    //   material->setObject("alphaTexture",value);
    //   return true;
    // }
    return false;
  }

}

