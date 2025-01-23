#pragma once

#include "rtcore/common/Backend.h"
#include "barney/common/cuda-helper.h"

namespace barney {
  namespace cuda {

    struct BaseBackend;
    struct DevGroup;
    
    struct SetActiveGPU {
      inline SetActiveGPU(const rtc::Device *device)
        : savedDevice(device)
      {
        if (!device) return;
        savedActiveDeviceID = device->setActive();
      }

      inline ~SetActiveGPU()
      {
        savedDevice->restoreActive(savedActiveDeviceID);
      }
    private:
      int savedActiveDeviceID = -1;
      const rtc::Device *const savedDevice;
    };
  

    struct BaseDevice : public rtc::Device {
      BaseDevice(int physicalGPU)
        : rtc::Device(physicalGPU)
      {
        int saved = setActive();
        cudaStreamCreate(&stream);
        restoreActive(saved);
      }
      
      void copyAsync(void *dst, void *src, size_t numBytes) override;
      void *allocHost(size_t numBytes) override;
      void freeHost(void *mem) override;
      void memsetAsync(void *mem,int value, size_t size) override;
      void *alloc(size_t numBytes) override;
      void free(void *mem) override;
      void sync() override;
      
      /*! sets this gpu as active, and returns physical ID of GPU that
        was active before */
      int setActive() const override;
      
      /*! restores the gpu whose ID was previously returend by setActive() */
      void restoreActive(int oldActive) const override;


      rtc::TextureData *
      createTextureData(vec3i dims,
                        rtc::DataType format,
                        const void *texels) override;

      void freeTextureData(rtc::TextureData *) override 
      { BARNEY_NYI(); };
      void freeTexture(rtc::Texture *) override 
      { BARNEY_NYI(); };
      
      cudaStream_t stream;
    };

    struct TextureData : public rtc::TextureData {
      TextureData(BaseDevice *device,
                  vec3i dims,
                  rtc::DataType format,
                  const void *texels);

      rtc::Texture *
      createTexture(const rtc::TextureDesc &desc) override;
      
      cudaArray_t array;
      cudaTextureReadMode readMode;
    };
    
    struct Texture : public rtc::Texture {
      Texture(TextureData *data,
              const rtc::TextureDesc &desc);
      
      rtc::device::TextureObject getDD() const override
      { return (const rtc::device::TextureObject&)textureObject; }
      
      cudaTextureObject_t textureObject;
    };
    
    struct BaseBackend : public rtc::Backend {
      BaseBackend();
    };
    
    struct CUDABackend : public cuda::BaseBackend {
      std::vector<rtc::Device *>
      createDevices(const std::vector<int> &gpuIDs) override;
    };
    
  }
}

  
  
