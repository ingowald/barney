#pragma once

#include "rtcore/common/Backend.h"
#include "rtcore/embree/Buffer.h"
#include "rtcore/embree/GeomType.h"

namespace barney {
  namespace embree {

    struct Geom : public rtc::Geom
    {
      Geom(GeomType *type);
      
      void setDD(const void *dd) override;
      
      std::vector<int8_t> programData;
      GeomType *const type;
    };
    
  }
}
