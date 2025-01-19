#pragma once

#include "rtcore/common/Backend.h"
#include "barney/common/cuda-helper.h"

namespace barney {
  namespace cuda {

    struct BaseBackend;
    struct DevGroup;
    struct BaseDevGroup;
    
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
      BaseDevice(int physicalGPU, int localID)
        : rtc::Device(physicalGPU,localID)
      {}
      
      void copyAsync(void *dst, void *src, size_t numBytes) override;
      void *allocHost(size_t numBytes) override;
      void freeHost(void *mem) override;
      void memsetAsync(void *mem,int value, size_t size) override;
      void *alloc(size_t numBytes) override;
      void free(void *mem) override;
      void buildPipeline() override;
      void buildSBT() override;
      void sync() override;
      
      /*! sets this gpu as active, and returns physical ID of GPU that
        was active before */
      int setActive() const override;
      
      /*! restores the gpu whose ID was previously returend by setActive() */
      void restoreActive(int oldActive) const override;
    };

    struct TextureData : public rtc::TextureData {
      TextureData(const BaseDevGroup *dg,
                  vec3i dims,
                  rtc::TextureData::Format format,
                  const void *texels);
      struct PerDev {
        cudaArray_t array;
      };
      std::vector<PerDev> perDev;
      cudaTextureReadMode readMode;
    };
    
    struct Texture : public rtc::Texture {
      Texture(const BaseDevGroup *dg,
              TextureData *data,
              // rtc::Texture::ReadMode   readMode,
              rtc::Texture::FilterMode filterMode,
              rtc::Texture::AddressMode addressModes[3],
              const vec4f borderColor,
              bool normalizedCoords,
              rtc::Texture::ColorSpace colorSpace);

      rtc::device::TextureObject getDD(const rtc::Device *) const override;
      
      std::vector<cudaTextureObject_t> perDev;
    };
    
    struct BaseDevGroup : public rtc::DevGroup {
      BaseDevGroup(BaseBackend *backend,
                   const std::vector<int> &gpuIDs);
      
      rtc::TextureData *
      createTextureData(vec3i dims,
                        rtc::TextureData::Format format,
                        const void *texels) const
        override;

      rtc::Texture *
      createTexture(rtc::TextureData *data,
                    // rtc::Texture::ReadMode   readMode,
                    rtc::Texture::FilterMode filterMode,
                    rtc::Texture::AddressMode addressModes[3],
                    const vec4f borderColor = vec4f(0.f),
                    bool normalizedCoords = true,
                    rtc::Texture::ColorSpace colorSpace
                    = rtc::Texture::COLOR_SPACE_LINEAR) const override;
      
      void freeTextureData(rtc::TextureData *) const
        override 
      { BARNEY_NYI(); };
      void freeTexture(rtc::Texture *) const 
        override 
      { BARNEY_NYI(); };


    };
    
    struct BaseBackend : public rtc::Backend {
      BaseBackend();
      // void setActiveGPU(int physicalID) override;
      // int  getActiveGPU() override;
    };
    
    struct CUDABackend : public cuda::BaseBackend {
      rtc::DevGroup *createDevGroup(const std::vector<int> &gpuIDs) override;
    };
    
  }
}

  
  
