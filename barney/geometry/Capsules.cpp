// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/geometry/Capsules.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"

namespace BARNEY_NS {

  RTC_IMPORT_USER_GEOM(Capsules,Capsules,Capsules::DD,false,false);

  Capsules::Capsules(Context *context, DevGroup::SP devices)
    : Geometry(context,devices)
  {}

  void Capsules::commit()
  {
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      if (pld->userGeoms.empty()) {
        rtc::GeomType *gt
          = device->geomTypes.get(createGeomType_Capsules);
        rtc::Geom *geom = gt->createGeom();
        pld->userGeoms = { geom };
      }
      rtc::Geom *geom = pld->userGeoms[0];
      
      assert(indices);
      int numIndices = indices ? (int)indices->count : 0;
      if (numIndices == 0)
        std::cout << OWL_TERMINAL_RED
                  << "#bn.capsules: warning - empty indices array"
                  << OWL_TERMINAL_DEFAULT
                  << std::endl;
      geom->setPrimCount(numIndices);

      Capsules::DD dd;
      Geometry::writeDD(dd,device);
      dd.vertices  = (vec4f*)(vertices?vertices->getDD(device):0);
      dd.indices   = (vec2i*)(indices?indices->getDD(device):0);
      // done:
      geom->setDD(&dd);
      
    }
  } 

  bool Capsules::setData(const std::string &member, const barney_api::Data::SP &value)
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
