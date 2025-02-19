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
      
      void copyAsync(void *dst, const void *src, size_t numBytes) override;
      void *allocHost(size_t numBytes) override;
      void freeHost(void *mem) override;
      void memsetAsync(void *mem,int value, size_t size) override;
      void *allocMem(size_t numBytes) override;
      void freeMem(void *mem) override;
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

      void freeTextureData(rtc::TextureData *) override;
      void freeTexture(rtc::Texture *) override;
      
      cudaStream_t stream;
    };

    struct TextureData : public rtc::TextureData {
      TextureData(BaseDevice *device,
                  vec3i dims,
                  rtc::DataType format,
                  const void *texels);
      virtual ~TextureData() override;
      
      rtc::Texture *
      createTexture(const rtc::TextureDesc &desc) override;
      
      cudaArray_t array;
      cudaTextureReadMode readMode;
    };
    
    struct Texture : public rtc::Texture {
      Texture(TextureData *data,
              const rtc::TextureDesc &desc);
      virtual ~Texture() override;
      
      rtc::device::TextureObject getDD() const override
      { return (const rtc::device::TextureObject&)textureObject; }
      
      cudaTextureObject_t textureObject;
    };
    
    struct BaseBackend : public rtc::Backend {
      BaseBackend();
    };
    
    struct CUDABackend : public cuda::BaseBackend {
      rtc::Device *createDevice(int gpuID) override;
    };
    
  }
}

  
  
