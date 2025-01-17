#include "rtcore/embree/EmbreeBackend.h"

namespace barney {
  namespace embree {
    
    EmbreeBackend::EmbreeBackend()
    {
      numPhysicalDevices = 1;
    }

    DevGroup::DevGroup(EmbreeBackend *backend,
                       const std::vector<int> &gpuIDs,
                       size_t sizeOfGlobals)
      : rtc::DevGroup(backend)
    {
    }
    
    rtc::DevGroup *EmbreeBackend
    ::createDevGroup(const std::vector<int> &gpuIDs,
                     size_t sizeOfGlobals)
    {
      return new DevGroup(this,gpuIDs,sizeOfGlobals);
    }
    
    
  }
  namespace rtc {
    Backend *createBackend_embree()
    {
      return new barney::embree::EmbreeBackend;
    }
  }
}

  
  


