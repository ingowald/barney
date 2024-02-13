// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

#include "barney/geometry/Spheres.h"
#include "barney/Model.h"

namespace barney {
  
  extern "C" char Spheres_ptx[];
  
  OWLGeomType Spheres::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'Spheres' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;
    
    std::vector<OWLVarDecl> params
      = {
         { "radii", OWL_BUFPTR, OWL_OFFSETOF(DD,radii) },
         { "defaultRadius", OWL_FLOAT, OWL_OFFSETOF(DD,defaultRadius) },
         { "origins", OWL_BUFPTR, OWL_OFFSETOF(DD,origins) },
         { "colors", OWL_BUFPTR, OWL_OFFSETOF(DD,colors) },
    };
    Geometry::addVars(params,0);
    OWLModule module = owlModuleCreate
      (devGroup->owl,Spheres_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(Spheres::DD),
       params.data(),(int)params.size());
    owlGeomTypeSetBoundsProg(gt,module,"SpheresBounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"SpheresIsec");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"SpheresCH");
    owlBuildPrograms(devGroup->owl);
    
    return gt;
  }
  
  Spheres::Spheres(DataGroup *owner)
    : Geometry(owner)
  {
    // originsBuffer = owlManagedMemoryBufferCreate
    //   (owner->devGroup->owl,OWL_FLOAT3,numOrigins,origins);
    // if (colors)
    //   colorsBuffer = owlManagedMemoryBufferCreate
    //     (owner->devGroup->owl,OWL_FLOAT3,numOrigins,colors);

  }

  void Spheres::commit()
  {
    if (userGeoms.empty()) {
      OWLGeomType gt = owner->devGroup->getOrCreateGeomTypeFor
        ("Spheres",Spheres::createGeomType);
      OWLGeom geom = owlGeomCreate(owner->devGroup->owl,gt);
      userGeoms.push_back(geom);
    }
    OWLGeom geom = userGeoms[0];
    
    Geometry::commit();
    owlGeomSet1f(geom,"defaultRadius",defaultRadius);
    owlGeomSetBuffer(geom,"origins",origins?origins->owl:0);
    owlGeomSetBuffer(geom,"colors",colors?colors->owl:0);
    int numOrigins = origins->count;
    owlGeomSetPrimCount(geom,numOrigins);
    material->set(geom);
  } 

  bool Spheres::set1f(const std::string &member, const float &value)
  {
    if (Geometry::set1f(member,value))
      return true;
    if (member == "radius") {
      defaultRadius = value;
      return true;
    }
    return false;
  }
  
  bool Spheres::setData(const std::string &member, const Data::SP &value)
  {
    if (Geometry::setData(member,value))
      return true;
    if (member == "colors") {
      colors = value->as<PODData>();
      return true;
    }
    if (member == "origins") {
      origins = value->as<PODData>();
      return true;
    }
    if (member == "radii") {
      radii = value->as<PODData>();
      return true;
    }
    return false;
  }

  bool Spheres::setObject(const std::string &member, const Object::SP &value)
  {
    return 0;
  }

}

