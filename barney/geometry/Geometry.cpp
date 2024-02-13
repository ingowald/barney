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
#include "barney/DataGroup.h"
#include "barney/Context.h"

#include "barney/geometry/Spheres.h"
#include "barney/geometry/Cylinders.h"

namespace barney {
  
  Geometry::SP Geometry::create(DataGroup *owner, const std::string &type)
  {
    if (type == "spheres")
      return std::make_shared<Spheres>(owner);
    if (type == "cylinders")
      return std::make_shared<Cylinders>(owner);
    
    owner->context->warn_unsupported_object("Geometry",type);
    return {};
  }

  Geometry::Geometry(DataGroup *owner)
    : Object(owner->context),
      owner(owner),
      material(std::make_shared<Material>(owner))
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
    Material::addVars(vars,base+OWL_OFFSETOF(DD,material));
  }

  // void Geometry::setMaterial(OWLGeom geom)
  // {
  //   material.set(geom);
  // }
  
}

