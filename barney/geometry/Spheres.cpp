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

#include "barney/geometry/Spheres.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"

RTC_IMPORT_USER_GEOM_TYPE(Spheres,BARNEY_NS::Spheres::DD,false,true);

namespace BARNEY_NS {

  // extern "C" char Spheres_ptx[];
  
  Spheres::Spheres(SlotContext *slotContext)
    : Geometry(slotContext)
  {}


  // rtc::GeomType *Spheres::createGeomType(rtc::Device *device, const void *)
  // {
  //   if (Context::logging())
  //     std::cout << OWL_TERMINAL_GREEN
  //               << "creating 'Spheres' geometry type"
  //               << OWL_TERMINAL_DEFAULT << std::endl;
  //   return device->createUserGeomType("Spheres_ptx",
  //                                     "Spheres",
  //                                     sizeof(Spheres::DD),
  //                                     /*ah*/false,/*ch*/true);
  // }
  
  void Spheres::commit()
  {
    if (!origins) return;

    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      auto rtc = device->rtc;
      if (pld->userGeoms.empty()) {
        int numOrigins = (int)origins->count;
        rtc::GeomType *gt
          = device->geomTypes.get("Spheres",Spheres::createGeomType);
        rtc::Geom *geom = gt->createGeom();
        geom->setPrimCount(numOrigins);
        pld->userGeoms.push_back(geom);
      }
      rtc::Geom *geom = pld->userGeoms[0];
      
      Spheres::DD dd;
      Geometry::writeDD(dd,device);
      dd.origins = (vec3f*)(origins->getDD(device));
      dd.radii   = (float*)(radii?radii->getDD(device):0);
      dd.colors  = (vec3f*)(colors?colors->getDD(device):0);
      dd.defaultRadius = defaultRadius;
      // done:
      geom->setDD(&dd);
    }
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
    // if (member == "vertex.attribute0") {
    //   vertexAttribute0 = value->as<PODData>();
    //   return true;
    // }
    // if (member == "vertex.attribute1") {
    //   vertexAttribute1 = value->as<PODData>();
    //   return true;
    // }
    // if (member == "vertex.attribute2") {
    //   vertexAttribute2 = value->as<PODData>();
    //   return true;
    // }
    // if (member == "vertex.attribute3") {
    //   vertexAttribute3 = value->as<PODData>();
    //   return true;
    // }
    // if (member == "vertex.attribute4") {
    //   vertexAttribute4 = value->as<PODData>();
    //   return true;
    // }
    return false;
  }

  bool Spheres::setObject(const std::string &member, const Object::SP &value)
  {
    if (Geometry::setObject(member,value))
      return true;
    return false;
  }

}

