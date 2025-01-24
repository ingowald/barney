#pragma once

#include "rtcore/embree/Device.h"

namespace barney {
  namespace embree {

    struct Buffer : public rtc::Buffer
    {
      Buffer(Device *device,size_t numBytes,const void *initMem);
      virtual ~Buffer();
      void *getDD() const override;
      void *mem = 0;
    };

  }
}
