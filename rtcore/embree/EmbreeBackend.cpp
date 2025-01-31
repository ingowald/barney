#include "rtcore/embree/Device.h"
#include "rtcore/embree/EmbreeBackend.h"

namespace barney {
  namespace embree {
    struct TextureData;
    struct Device;

    EmbreeBackend::EmbreeBackend()
    {
      numPhysicalDevices = 1;
    }

    EmbreeBackend::~EmbreeBackend()
    {}
    
    rtc::Device *EmbreeBackend::createDevice(int gpuID)
    {
      return new embree::Device(gpuID);
    }
    
  }
  namespace rtc {
    
      extern "C"
    Backend *createBackend_embree()
    {
      return new barney::embree::EmbreeBackend;
    }
  }
}

  
  


