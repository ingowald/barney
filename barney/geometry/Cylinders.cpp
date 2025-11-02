// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/geometry/Cylinders.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"

namespace BARNEY_NS {

  RTC_IMPORT_USER_GEOM(Cylinders,Cylinders,Cylinders::DD,false,false);

  Cylinders::Cylinders(Context *context, DevGroup::SP devices)
    : Geometry(context,devices)
  {}

  void Cylinders::commit()
  {
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      if (pld->userGeoms.empty()) {
        rtc::GeomType *gt
          = device->geomTypes.get(createGeomType_Cylinders);
        rtc::Geom *geom = gt->createGeom();
        geom->setPrimCount((int)indices->count);
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
                          const barney_api::Data::SP &value)
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

