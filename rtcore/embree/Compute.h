#pragma once

#include "rtcore/embree/Device.h"

namespace barney {
  namespace embree {

    struct ComputeInterface {
    };
    
    struct Compute : public rtc::Compute
    {
      typedef void (*ComputeFct)(ComputeInterface &,
                                 const void *dd);
      
      Compute(Device *device, const std::string &name);
      ComputeFct computeFct = 0;
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
