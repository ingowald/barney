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
#include "barney/Context.h"

namespace barney {

  extern "C" char Capsules_ptx[];

  Capsules::Capsules(SlotContext *slotContext)
    : Geometry(slotContext)
  {}

  rtc::GeomType *Capsules::createGeomType(rtc::Device *device)
  {
    if (Context::logging())
      std::cout << OWL_TERMINAL_GREEN
                << "creating 'Capsules' geometry type"
                << OWL_TERMINAL_DEFAULT << std::endl;

    return device->createUserGeomType("Capsules",
                                      sizeof(Capsules::DD),
                                      /*ah*/false,/*ch*/true);
                                      // "CapsulesBounds",
                                      // "CapsulesIsec",
                                      // nullptr,
                                      // "CapsulesCH");
  }
  
  void Capsules::commit()
  {
    for (auto device : *devices) {
      auto rtc = device->rtc;
      PLD *pld = getPLD(device);
      // if (userGeoms.empty()) {
      //   OWLGeomType gt = getDevGroup()->getOrCreateGeomTypeFor
      //     ("Capsules",Capsules::createGeomType);
      //   OWLGeom geom = owlGeomCreate(getDevGroup()->owl,gt);
      //   userGeoms.push_back(geom);
      // }
      if (pld->userGeoms.empty()) {
        rtc::GeomType *gt
          = device->geomTypes.get("Capsules",
                                  Capsules::createGeomType);
        rtc::Geom *geom = gt->createGeom();
        pld->userGeoms = { geom };
      }
      // OWLGeom geom = userGeoms[0];
      rtc::Geom *geom = pld->userGeoms[0];

      Capsules::DD dd;
      Geometry::writeDD(dd,device);
      dd.vertices  = (vec4f*)(vertices?vertices->getDD(device):0);
      dd.indices   = (vec2i*)(indices?indices->getDD(device):0);
      // done:
      geom->setDD(&dd);
      
      // Geometry::commit();
      
      // owlGeomSetBuffer(geom,"vertices",vertices?vertices->owl:0);
      // owlGeomSetBuffer(geom,"indices",indices?indices->owl:0);
      assert(indices);
      int numIndices = indices ? indices->count : 0;
      if (numIndices == 0)
        std::cout << OWL_TERMINAL_RED
                  << "#bn.capsules: warning - empty indices array"
                  << OWL_TERMINAL_DEFAULT
                  << std::endl;
      geom->setPrimCount(numIndices);
      // owlGeomSetPrimCount(geom,numIndices);
      
    //   setAttributesOn(geom);
    // getMaterial()->setDeviceDataOn(geom);
    }
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

