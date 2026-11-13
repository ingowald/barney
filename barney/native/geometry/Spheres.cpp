// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/geometry/Spheres.h"
#include "native/ModelSlot.h"
#include "native/Context.h"

namespace BARNEY_NS {
  namespace native {
    
    RTC_IMPORT_USER_GEOM(Spheres,Spheres,Spheres::DD,false,true);
  
    Spheres::Spheres(Device *device)
      : Geometry(device)
    {}

    void Spheres::commit()
    {
      if (!origins) return;

      auto pld = &this->rtc;
        if (pld->userGeoms.empty()) {
          int numOrigins = (int)origins->count;
          rtc::GeomType *gt
            = device->geomTypes.get(createGeomType_Spheres);
          rtc::Geom *geom = gt->createGeom();
          geom->setPrimCount(numOrigins);
          pld->userGeoms.push_back(geom);
        }
        rtc::Geom *geom = pld->userGeoms[0];
      
        Spheres::DD dd;
        Geometry::writeDD(dd);
        dd.origins = (vec3f*)(origins->getDD());
        dd.radii   = (float*)(radii?radii->getDD():0);
        dd.colors  = (vec3f*)(colors?colors->getDD():0);
        dd.defaultRadius = defaultRadius;
        // done:
        geom->setDD(&dd);
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
      if (Geometry::setObject(member,value))
        return true;
      return false;
    }

  }
}

