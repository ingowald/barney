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
#include "barney/Context.h"

namespace barney {

  extern "C" char Cylinders_ptx[];

  Cylinders::Cylinders(SlotContext *slotContext)
    : Geometry(slotContext)
  {}

  rtc::GeomType *Cylinders::createGeomType(rtc::Device *device)
  // OWLGeomType Cylinders::createGeomType(DevGroup *devGroup)
  {
    if (Context::logging())
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'Cylinders' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;

    return device->createUserGeomType("Cylinders",
                                      sizeof(Cylinders::DD),
                                      /*ah*/false,/*ch*/true);
  }
  
  void Cylinders::commit()
  {
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      if (pld->userGeoms.empty()) {
        rtc::GeomType *gt
          = device->geomTypes.get("Cylinders",Cylinders::createGeomType);
        rtc::Geom *geom = gt->createGeom();
        geom->setPrimCount(indices->count);
        pld->userGeoms.push_back(geom);
      }
      rtc::Geom *geom = pld->userGeoms[0];

      Cylinders::DD dd;
      Geometry::writeDD(dd,device);
      dd.vertices = (vec3f*)(vertices?vertices->getDD(device):0);
      dd.indices  = (vec2i*)(indices?indices->getDD(device):0);
      dd.radii    = (float*)(radii?radii->getDD(device):0);
      geom->setDD(&dd);
    }
  } 

  bool Cylinders::set1i(const std::string &member,
                        const int &value)
  {
    if (Geometry::set1i(member,value))
      return true;
    return false;
  }
  
  bool Cylinders::set1f(const std::string &member,
                        const float &value)
  {
    if (Geometry::set1f(member,value))
      return true;
    return false;
  }
  
  bool Cylinders::setData(const std::string &member,
                          const Data::SP &value)
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

  bool Cylinders::setObject(const std::string &member,
                            const Object::SP &value)
  {
    if (Geometry::setObject(member,value))
      return true;
    return false;
  }

}

