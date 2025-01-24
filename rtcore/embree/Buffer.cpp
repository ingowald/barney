#include "rtcore/embree/Buffer.h"

namespace barney {
  namespace embree {

    Buffer::Buffer(Device *device,
                   size_t numBytes,
                   const void *initMem)
      : rtc::Buffer(device)
    {
      mem = malloc(numBytes);
      if (initMem)
        memcpy(mem,initMem,numBytes);
    }
    
    Buffer::~Buffer()
    {
      if (mem) free(mem);
    }
    
    void *Buffer::getDD() const
    {
      return mem;
    }

  }
}
