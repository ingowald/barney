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

#include "barney/geometry/Cylinders.h"
#include "barney/ModelSlot.h"

namespace barney {

  extern "C" char Cylinders_ptx[];

  Cylinders::Cylinders(ModelSlot *owner)
    : Geometry(owner)
  {}

  OWLGeomType Cylinders::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'Cylinders' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;
    
    std::vector<OWLVarDecl> params
      = {
         { "radii", OWL_BUFPTR, OWL_OFFSETOF(DD,radii) },
         { "vertices", OWL_BUFPTR, OWL_OFFSETOF(DD,vertices) },
         { "colors", OWL_BUFPTR, OWL_OFFSETOF(DD,colors) },
         { "indices", OWL_BUFPTR, OWL_OFFSETOF(DD,indices) },
         { "colorPerVertex", OWL_INT, OWL_OFFSETOF(DD,colorPerVertex) },
         { "radiusPerVertex", OWL_INT, OWL_OFFSETOF(DD,radiusPerVertex) },
    };
    Geometry::addVars(params,0);
    OWLModule module = owlModuleCreate
      (devGroup->owl,Cylinders_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(Cylinders::DD),
       params.data(), (int)params.size());
    owlGeomTypeSetBoundsProg(gt,module,"CylindersBounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"CylindersIsec");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"CylindersCH");
    owlBuildPrograms(devGroup->owl);
    
    return gt;
  }
  
  void Cylinders::commit()
  {
    if (userGeoms.empty()) {
      OWLGeomType gt = owner->devGroup->getOrCreateGeomTypeFor
        ("Cylinders",Cylinders::createGeomType);
      OWLGeom geom = owlGeomCreate(owner->devGroup->owl,gt);
      userGeoms.push_back(geom);
    }
    OWLGeom geom = userGeoms[0];
    
    Geometry::commit();
    owlGeomSet1i(geom,"colorPerVertex",colorPerVertex);
    owlGeomSet1i(geom,"radiusPerVertex",radiusPerVertex);
    owlGeomSetBuffer(geom,"vertices",vertices?vertices->owl:0);
    owlGeomSetBuffer(geom,"indices",indices?indices->owl:0);
    owlGeomSetBuffer(geom,"colors",colors?colors->owl:0);
    owlGeomSetBuffer(geom,"radii",radii?radii->owl:0);
    int numIndices = indices->count;
    PRINT(numIndices);
    owlGeomSetPrimCount(geom,numIndices);
    material->setDeviceDataOn(geom);
  } 

  bool Cylinders::set1i(const std::string &member, const int &value)
  {
    if (Geometry::set1i(member,value))
      return true;
    if (member == "radiusPerVertex") {
      radiusPerVertex = value;
      return true;
    }
    if (member == "colorPerVertex") {
      colorPerVertex = value;
      return true;
    }
    return false;
  }
  
  bool Cylinders::set1f(const std::string &member, const float &value)
  {
    if (Geometry::set1f(member,value))
      return true;
    return false;
  }
  
  bool Cylinders::setData(const std::string &member, const Data::SP &value)
  {
    if (Geometry::setData(member,value))
      return true;
    if (member == "colors") {
      colors = value->as<PODData>();
      return true;
    }
    if (member == "vertices") {
      vertices = value->as<PODData>();
      return true;
    }
    if (member == "indices") {
      indices = value->as<PODData>();
      return true;
    }
    if (member == "radii") {
      radii = value->as<PODData>();
      return true;
    }
    return false;
  }

  bool Cylinders::setObject(const std::string &member, const Object::SP &value)
  {
    if (Geometry::setObject(member,value))
      return true;
    return false;
  }

}

