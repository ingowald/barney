#include "rtcore/embree/Device.h"
#include "rtcore/embree/EmbreeBackend.h"

namespace barney {
  namespace embree {
    struct TextureData;
    struct Device;

    // struct Compute : public rtc::Compute
    // {
    //   Compute(Device *device, const std::string &name);
    // };
      
    
    

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
    
    Backend *createBackend_embree()
    {
      return new barney::embree::EmbreeBackend;
    }
  }
}

  
  


