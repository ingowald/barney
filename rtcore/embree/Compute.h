#pragma once

#include "rtcore/embree/Device.h"

namespace barney {
  namespace embree {

    struct ComputeInterface;
    struct TraceInterface;
    
    struct Compute : public rtc::Compute
    {
      typedef void (*ComputeFct)(ComputeInterface &,
                                 const void *dd);

      Compute(Device *device, const std::string &name);

      void launch(int numBlocks,
                  int blockSize,
                  const void *dd) override;
      
      void launch(vec2i numBlocks,
                  vec2i blockSize,
                  const void *dd) override;
      
      void launch(vec3i numBlocks,
                  vec3i blockSize,
                          const void *dd) override;
      
      ComputeFct computeFct = 0;
    };
    
    struct Trace : public rtc::Trace
    {
      typedef void (*TraceFct)(TraceInterface &,
                                 const void *dd);
      Trace(Device *device, const std::string &name, size_t sizeOfRG);
      
      void launch(vec2i launchDims,
                  const void *dd);      
      void launch(int launchDims,
                  const void *dd);
      
      void sync() override
      { /* no-op */ }
      TraceFct traceFct = 0;
    };

  }
}
