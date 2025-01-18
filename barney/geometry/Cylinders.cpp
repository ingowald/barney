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

  Cylinders::Cylinders(Context *context, int slot)
    : Geometry(context,slot)
  {}

  rtc::GeomType *Cylinders::createGeomType(DevGroup *devGroup)
  // OWLGeomType Cylinders::createGeomType(DevGroup *devGroup)
  {
    if (DevGroup::logging())
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'Cylinders' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;

#if 1
    return devGroup->rtc->createUserGeomType("Cylinders",
                                        sizeof(Cylinders::DD),
                                        "CylindersBounds",
                                        "CylindersIsec",
                                        "CylindersCH",
                                        nullptr/*AH*/);
                                        
#else
    // std::vector<OWLVarDecl> params
    //   = {
    //      { "radii", OWL_BUFPTR, OWL_OFFSETOF(DD,radii) },
    //      { "vertices", OWL_BUFPTR, OWL_OFFSETOF(DD,vertices) },
    //      { "indices", OWL_BUFPTR, OWL_OFFSETOF(DD,indices) },
    // };
    // Geometry::addVars(params,0);
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
#endif
  }
  
  void Cylinders::commit()
  {
    if (userGeoms.empty()) {
      // OWLGeomType gt = getDevGroup()->getOrCreateGeomTypeFor
      //   ("Cylinders",Cylinders::createGeomType);
      rtc::GeomType *gt = getDevGroup()->getOrCreateGeomTypeFor
        ("Cylinders",Cylinders::createGeomType);
      // OWLGeom geom = owlGeomCreate(getDevGroup()->owl,gt);
      rtc::Geom *geom = getRTC()->createGeom(gt);
      geom->setPrimCount(indices->count);
      userGeoms.push_back(geom);
    }
    // OWLGeom geom = userGeoms[0];
    rtc::Geom *geom = userGeoms[0];

#if 1
    DD dd;
    for (auto device : getRTC()->devices) {
      Geometry::writeDD(dd,device);
      dd.vertices = (vec3f*)getDD(device,vertices);
        // = vertices
        // ? vertices->getDD(device)
        // : nullptr;
      dd.indices  = (vec2i*)getDD(device,indices);
        // ? indices->getDD(device)
        // : nullptr;
      dd.radii = (float*)getDD(device,radii);
      //   = radii
      //   ? radii->getDD(device)
      //   : nullptr;
      // Geometry::setAttributes(dd,device);
      // setAttributes(dd,device);
      // dd.material = getMaterial()->getDD(device);
      geom->setDD(&dd,device);
    }
    
#else
    Geometry::commit();
      
    owlGeomSetBuffer(geom,"vertices",vertices?vertices->owl:0);
    owlGeomSetBuffer(geom,"indices",indices?indices->owl:0);
    owlGeomSetBuffer(geom,"radii",radii?radii->owl:0);
    assert(indices);
    int numIndices = indices->count;
    if (numIndices == 0)
      std::cout << OWL_TERMINAL_RED
                << "#bn.cylinders: warning - empty indices array"
                << OWL_TERMINAL_DEFAULT
                << std::endl;
    owlGeomSetPrimCount(geom,numIndices);
    
    setAttributesOn(geom);
    getMaterial()->setDeviceDataOn(geom);
#endif
  } 

  bool Cylinders::set1i(const std::string &member, const int &value)
  {
    if (Geometry::set1i(member,value))
      return true;
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

