#pragma once

#include "rtcore/common/rtcore-common.h"
#include <cuda_runtime.h>
// #include <cuda.h>
#ifdef __CUDACC__
#include <cuda/std/limits>
#endif
#include "cuda-helper.h"

// namespace barney {
namespace rtc {
  namespace cuda_common {

    using ::float2;
    using ::float3;
    using ::float4;
    using ::int2;
    using ::int3;
    using ::int4;
    
    inline __both__ vec4f load(const float4 &v)
    { return vec4f(v.x,v.y,v.z,v.w); }
    
    // struct BaseBackend;
    // struct DevGroup;
    
    struct Device;
    struct Texture;
    struct TextureData;
    
    struct SetActiveGPU {
      inline SetActiveGPU(const Device *device);
      inline ~SetActiveGPU();
    private:
      int savedActiveDeviceID = -1;
      const Device *const savedDevice;
    };
  
    
    struct ComputeKernel1D {
      virtual void launch(unsigned int nb, unsigned int bs,
                          const void *pKernelData) = 0;
    };
    struct ComputeKernel2D {
      virtual void launch(vec2ui nb, vec2ui bs,
                          const void *pKernelData) = 0;
      inline void launch(vec2i nb, vec2i bs,
                         const void *pKernelData)
      { launch(vec2ui(nb),vec2ui(bs),pKernelData); }
    };
    struct ComputeKernel3D {
      virtual void launch(vec3ui nb, vec3ui bs,
                          const void *pKernelData) = 0;
      inline void launch(vec3i nb, vec3i bs,
                         const void *pKernelData)
      { launch(vec3ui(nb),vec3ui(bs),pKernelData); }
    };
    
    struct Device {
      Device(int physicalGPU)
        : physicalID(physicalGPU)
      {
        int saved = setActive();
        cudaStreamCreate(&stream);
        restoreActive(saved);
      }
      
      void copyAsync(void *dst, const void *src, size_t numBytes);
      void copy(void *dst, const void *src, size_t numBytes)
      { copyAsync(dst,src,numBytes); sync(); }
      void *allocHost(size_t numBytes);
      void freeHost(void *mem);
      void memsetAsync(void *mem,int value, size_t size);
      void *allocMem(size_t numBytes);
      void freeMem(void *mem);
      void sync();
      
      /*! sets this gpu as active, and returns physical ID of GPU that
        was active before */
      int setActive() const;
      
      /*! restores the gpu whose ID was previously returend by setActive() */
      void restoreActive(int oldActive) const;

      TextureData *createTextureData(vec3i dims,
                                     rtc::DataType format,
                                     const void *texels);
      
      void freeTextureData(TextureData *);
      void freeTexture(Texture *);
      
      cudaStream_t stream;
      int const physicalID;
    };

    struct TextureData// : public rtc::TextureData
    {
      TextureData(Device *device,
                  vec3i dims,
                  rtc::DataType format,
                  const void *texels);
      virtual ~TextureData();
      
      Texture *
      createTexture(const rtc::TextureDesc &desc);
      
      cudaArray_t array;
      cudaTextureReadMode readMode;
      const vec3i dims;
      const DataType format;
      Device *const device;
    };
    
    struct Texture // : public rtc::Texture
    {
      Texture(TextureData *data,
              const TextureDesc &desc);
      virtual ~Texture();
      
      rtc::device::TextureObject getDD() const
      { return (const rtc::device::TextureObject&)textureObject; }

      TextureData *const data;
      cudaTextureObject_t textureObject;
    };
    
    // struct BaseBackend : public rtc::Backend {
    //   BaseBackend();
    // };
    
    // struct CUDABackend : public cuda::BaseBackend {
    //   rtc::Device *createDevice(int gpuID);
    // };

    inline SetActiveGPU::SetActiveGPU(const Device *device)
      : savedDevice(device)
    {
      if (!device) return;
      savedActiveDeviceID = device->setActive();
    }

    inline SetActiveGPU::~SetActiveGPU()
    {
      savedDevice->restoreActive(savedActiveDeviceID);
    }
    
  }
}

  
  
