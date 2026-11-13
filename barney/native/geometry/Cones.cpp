// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/geometry/Cones.h"
#include "native/ModelSlot.h"
#include "native/Context.h"

namespace BARNEY_NS {
  namespace native {
    
    RTC_IMPORT_USER_GEOM(Cones,Cones,Cones::DD,false,false);

    Cones::Cones(Device *device)
      : Geometry(device)
    {}

    std::string Cones::toString() const 
    { return "Cones{}"; }
    
    void Cones::commit()
    {
      if (!vertices || vertices->count == 0) {
        std::cout << OWL_TERMINAL_RED
                  << "#bn.cones: warning - empty vertices array"
                  << OWL_TERMINAL_DEFAULT
                  << std::endl;
        return;
      }
      auto pld = &this->rtc;
        if (pld->userGeoms.empty()) {
          rtc::GeomType *gt
            = device->geomTypes.get(createGeomType_Cones);
          rtc::Geom *geom = gt->createGeom();
          pld->userGeoms = { geom };
        }
        rtc::Geom *geom = pld->userGeoms[0];

        size_t numCones
          = indices
          ? indices->count
          : (vertices->count/2);
        assert(vertices);
        geom->setPrimCount((int)numCones);

        Cones::DD dd;
        Geometry::writeDD(dd);
        dd.vertices  = (vec3f*)(vertices?vertices->getDD():0);
        dd.indices   = (vec2i*)(indices?indices->getDD():0);
        dd.radii     = (float*)(radii?radii->getDD():0);
        // done:
        geom->setDD(&dd);
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
}


