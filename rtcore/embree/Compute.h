#pragma once

#include "rtcore/embree/Device.h"

namespace barney {
  namespace embree {
    
    struct Compute : public rtc::Compute
    {
      Compute(Device *device, const std::string &name);
    };
    
    struct Trace : public rtc::Trace
    {
      Trace(Device *device, const std::string &name, size_t sizeOfRG);
      void launch(vec2i launchDims,
                  const void *dd) override
      { BARNEY_NYI(); }
      
      void launch(int launchDims,
                  const void *dd) override
      { BARNEY_NYI(); }
      
      void sync() override
      { /* no-op */ }
    };

  }
}
