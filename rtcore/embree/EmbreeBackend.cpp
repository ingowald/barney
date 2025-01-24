#include "rtcore/embree/Device.h"
#include "rtcore/embree/EmbreeBackend.h"

namespace barney {
  namespace embree {
    struct TextureData;
    struct Device;

    struct SWTexture;
    
    struct Texture : public rtc::Texture
    {
      Texture(TextureData *const data,
              const rtc::TextureDesc &desc);
      rtc::device::TextureObject getDD() const override
      { return (const rtc::device::TextureObject &)swTex; }

      SWTexture *swTex = 0;
    };
      
    struct TextureData : public rtc::TextureData
    {
      TextureData(Device *device,
                  vec3i dims,
                  rtc::DataType format,
                  const void *texels);
      rtc::Texture *createTexture(const rtc::TextureDesc &desc) override
        
      { return new Texture(this,desc); }
      
      Device *const device;
    };
      
    struct Buffer : public rtc::Buffer
    {
      Buffer(Device *device,size_t numBytes,const void *initMem);
      virtual ~Buffer();
      void *getDD() const override;
      void *mem = 0;
    };
      
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
    

    Texture::Texture(TextureData *const data,
                     const rtc::TextureDesc &desc)
      : rtc::Texture(data,desc)
    {
      BARNEY_NYI();
    }

      TextureData::TextureData(Device *device,
                  vec3i dims,
                  rtc::DataType format,
                  const void *texels)
        : rtc::TextureData(device,dims,format),
          device(device)
      { BARNEY_NYI();}


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


    Compute::Compute(Device *device,
                     const std::string &name)
      : rtc::Compute(device)
    { BARNEY_NYI(); }
    
    Trace::Trace(Device *device,
                 const std::string &name,
                 size_t sizeOfRG)
      : rtc::Trace(device)
    { BARNEY_NYI(); }
    
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
  namespace rtc {
    
    Backend *createBackend_embree()
    {
      return new barney::embree::EmbreeBackend;
    }
  }
}

  
  


