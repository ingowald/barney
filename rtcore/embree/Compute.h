#pragma once

#include "rtcore/embree/Device.h"

namespace barney {
  namespace embree {
    
    struct Compute : public rtc::Compute
    {
      Compute(Device *device, const std::string &name);
    };
    
  }
}
