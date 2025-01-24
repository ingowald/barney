#include "rtcore/embree/Device.h"
#include "rtcore/embree/Compute.h"
#include "rtcore/embree/Texture.h"
#include "rtcore/embree/Buffer.h"

namespace barney {
  namespace embree {

    Device::Device(int physicalGPU)
      : rtc::Device(physicalGPU)
    {
      embreeDevice = rtcNewDevice("verbose=0");
    }

    Device::~Device()
    {
      destroy();
    }

    void Device::destroy()
    {
      rtcReleaseDevice(embreeDevice);
      embreeDevice = 0;
    }


    
    rtc::TextureData *Device::createTextureData(vec3i dims,
                                                rtc::DataType format,
                                                const void *texels) 
    { return new TextureData(this,dims,format,texels); }

    rtc::Buffer *Device::createBuffer(size_t numBytes,
                                      const void *initValues) 
    {
      return new Buffer(this,numBytes,initValues);
    }

    void Device::freeTextureData(rtc::TextureData *td) 
    { delete td; }
      
    void Device::freeTexture(rtc::Texture *tex) 
    { delete tex; }


    void Device::freeBuffer(rtc::Buffer *buffer) 
    {
      delete buffer;
    }
    
    rtc::Compute *Device::createCompute(const std::string &name) 
    { return new Compute(this,name); }
    
    
    rtc::Trace *Device::createTrace(const std::string &name,
                                    size_t rayGenSize) 
    { return new Trace(this,name,rayGenSize); }
    
    
  }
}

