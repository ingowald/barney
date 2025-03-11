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
#include "barney/Context.h"

namespace BARNEY_NS {

  RTC_IMPORT_USER_GEOM(Cones,Cones,Cones::DD,false,false);

  Cones::Cones(Context *context, DevGroup::SP devices)
    : Geometry(context,devices)
  {}

  void Cones::commit()
  {
    if (!vertices || vertices->count == 0) {
      std::cout << OWL_TERMINAL_RED
                << "#bn.cones: warning - empty vertices array"
                << OWL_TERMINAL_DEFAULT
                << std::endl;
      return;
    }
    for (auto device : *devices) {
      auto rtc = device->rtc;
      PLD *pld = getPLD(device);
      if (pld->userGeoms.empty()) {
        rtc::GeomType *gt
          = device->geomTypes.get(createGeomType_Cones);
        rtc::Geom *geom = gt->createGeom();
        pld->userGeoms = { geom };
      }
      rtc::Geom *geom = pld->userGeoms[0];

      int numCones
        = indices
        ? indices->count
        : (vertices->count/2);
      assert(vertices);
      geom->setPrimCount(numCones);

      Cones::DD dd;
      Geometry::writeDD(dd,device);
      dd.vertices  = (vec3f*)(vertices?vertices->getDD(device):0);
      dd.indices   = (vec2i*)(indices?indices->getDD(device):0);
      dd.radii     = (float*)(radii?radii->getDD(device):0);
      // done:
      geom->setDD(&dd);
    }
  } 
  
  bool Cones::setData(const std::string &member, const Data::SP &value)
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

}


