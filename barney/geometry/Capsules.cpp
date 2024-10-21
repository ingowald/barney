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

#include "barney/geometry/Capsules.h"
#include "barney/ModelSlot.h"

namespace barney {

  extern "C" char Capsules_ptx[];

  Capsules::Capsules(ModelSlot *owner)
    : Geometry(owner)
  {}

  OWLGeomType Capsules::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'Capsules' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;
    
    std::vector<OWLVarDecl> params
      = {
      { "vertices", OWL_BUFPTR, OWL_OFFSETOF(DD,vertices) },
      { "indices",  OWL_BUFPTR, OWL_OFFSETOF(DD,indices)  } 
    };
    Geometry::addVars(params,0);
    OWLModule module = owlModuleCreate
      (devGroup->owl,Capsules_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(Capsules::DD),
       params.data(), (int)params.size());
    owlGeomTypeSetBoundsProg(gt,module,"CapsulesBounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"CapsulesIsec");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"CapsulesCH");
    owlBuildPrograms(devGroup->owl);
    
    return gt;
  }
  
  void Capsules::commit()
  {
    if (userGeoms.empty()) {
      OWLGeomType gt = owner->devGroup->getOrCreateGeomTypeFor
        ("Capsules",Capsules::createGeomType);
      OWLGeom geom = owlGeomCreate(owner->devGroup->owl,gt);
      userGeoms.push_back(geom);
    }
    OWLGeom geom = userGeoms[0];

    Geometry::commit();

    owlGeomSetBuffer(geom,"vertices",vertices?vertices->owl:0);
    owlGeomSetBuffer(geom,"indices",indices?indices->owl:0);
    assert(indices);
    int numIndices = indices->count;
    if (numIndices == 0)
      std::cout << OWL_TERMINAL_RED
                << "#bn.capsules: warning - empty indices array"
                << OWL_TERMINAL_DEFAULT
                << std::endl;

    owlGeomSetPrimCount(geom,numIndices);
    
    setAttributesOn(geom);
    getMaterial()->setDeviceDataOn(geom);
  } 

  bool Capsules::setData(const std::string &member, const Data::SP &value)
  {
    if (Geometry::setData(member,value))
      // does 'primitive.color', and all the 'attributeN's
      return true;
    
    if (member == "vertices") {
      vertices = value->as<PODData>();
      return true;
    }
    if (member == "indices") {
      indices = value->as<PODData>();
      return true;
    }
    return false;
  }

}

