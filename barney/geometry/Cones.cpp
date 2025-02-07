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

#include "barney/geometry/Cones.h"
#include "barney/ModelSlot.h"

namespace barney {

  extern "C" char Cones_ptx[];

  Cones::Cones(ModelSlot *owner)
    : Geometry(owner)
  {}

  OWLGeomType Cones::createGeomType(DevGroup *devGroup)
  {
    if (DevGroup::logging())
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'Cones' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;
    
    std::vector<OWLVarDecl> params
      = {
      { "radii", OWL_BUFPTR, OWL_OFFSETOF(DD,radii) },
      { "vertices", OWL_BUFPTR, OWL_OFFSETOF(DD,vertices) },
      { "indices", OWL_BUFPTR, OWL_OFFSETOF(DD,indices) },
    };
    Geometry::addVars(params,0);
    OWLModule module = owlModuleCreate
      (devGroup->owl,Cones_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(Cones::DD),
       params.data(), (int)params.size());
    owlGeomTypeSetBoundsProg(gt,module,"ConesBounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"ConesIsec");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"ConesCH");
    owlBuildPrograms(devGroup->owl);
    
    return gt;
  }
  
  void Cones::commit()
  {
    if (userGeoms.empty()) {
      OWLGeomType gt = getDevGroup()->getOrCreateGeomTypeFor
        ("Cones",Cones::createGeomType);
      OWLGeom geom = owlGeomCreate(getDevGroup()->owl,gt);
      userGeoms.push_back(geom);
    }
    OWLGeom geom = userGeoms[0];
    
    Geometry::commit();
    owlGeomSetBuffer(geom,"vertices",vertices?vertices->owl:0);
    owlGeomSetBuffer(geom,"indices",indices?indices->owl:0);
    owlGeomSetBuffer(geom,"radii",radii?radii->owl:0);
    int numIndices = indices->count;
    owlGeomSetPrimCount(geom,numIndices);
    material->setDeviceDataOn(geom);
  } 
  
  bool Cones::setData(const std::string &member, const Data::SP &value)
  {
    if (Geometry::setData(member,value))
      return true;
    if (member == "colors") {
      colors = value->as<PODData>();
      return true;
    }
    if (member == "vertices") {
      vertices = value->as<PODData>();
      PRINT(vertices->count);
      return true;
    }
    if (member == "indices") {
      indices = value->as<PODData>();
      PRINT(indices);
      PRINT(indices->count);
      return true;
    }
    if (member == "radii") {
      radii = value->as<PODData>();
      PRINT(radii->count);
      return true;
    }
    return false;
  }

}


