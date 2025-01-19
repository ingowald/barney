#include "rtcore/cuda/CUDABackend.h"

namespace barney {
  namespace cuda {

    BaseBackend::BaseBackend()
    {
      cudaFree(0);
      BARNEY_CUDA_CALL(GetDeviceCount(&numPhysicalDevices));
    }
    
    BaseDevGroup::BaseDevGroup(BaseBackend *backend,
                               const std::vector<int> &gpuIDs)
      : rtc::DevGroup(backend)
    {}
    
    int BaseDevice::setActive() const
    {
      int oldActive = 0;
      BARNEY_CUDA_CHECK(cudaGetDevice(&oldActive));
      BARNEY_CUDA_CHECK(cudaSetDevice(physicalID));
      return oldActive;
    }
    
    void BaseDevice::restoreActive(int oldActive) const
    {
      BARNEY_CUDA_CHECK(cudaSetDevice(oldActive));
    }
    
    void *BaseDevice::alloc(size_t numBytes)
    {
      void *ptr = 0;
      BARNEY_CUDA_CALL(Malloc((void **)&ptr,numBytes));
      return ptr;
    }
    
    void *BaseDevice::allocHost(size_t numBytes) 
    {
      void *ptr = 0;
      BARNEY_CUDA_CALL(MallocHost(&ptr,numBytes));
      return ptr;
    }
      
    void BaseDevice::freeHost(void *mem) 
    {
      BARNEY_CUDA_CALL(FreeHost(mem));
    }
      
    void BaseDevice::free(void *mem) 
    {
      BARNEY_CUDA_CALL(Free(mem));
    }
      
    void BaseDevice::memsetAsync(void *mem,int value, size_t size) 
    {
      BARNEY_NYI();
    }
      

    void BaseDevice::copyAsync(void *dst, void *src, size_t numBytes) 
    {
      BARNEY_NYI();
    }
      
    void BaseDevice::buildPipeline() 
    {
      BARNEY_NYI();
    }
      
    void BaseDevice::buildSBT() 
    {
      BARNEY_NYI();
    }
      
    void BaseDevice::sync() 
    {
      BARNEY_NYI();
    }
      

    TextureData::TextureData(const BaseDevGroup *dg,
                             vec3i dims,
                             rtc::TextureData::Format format,
                             const void *texels)
      : rtc::TextureData(dims,format)
    {
      cudaChannelFormatDesc desc;
      size_t sizeOfScalar;
      size_t numScalarsPerTexel;
      switch (format) {
      case TextureData::FLOAT:
        desc         = cudaCreateChannelDesc<float>();
        sizeOfScalar = 4;
        readMode     = cudaReadModeElementType;
        numScalarsPerTexel = 1;
        break;
      case TextureData::FLOAT4:
        desc         = cudaCreateChannelDesc<float4>();
        sizeOfScalar = 4;
        readMode     = cudaReadModeElementType;
        numScalarsPerTexel = 4;
        break;
      case TextureData::UCHAR:
        desc         = cudaCreateChannelDesc<uint8_t>();
        sizeOfScalar = 1;
         readMode     = cudaReadModeNormalizedFloat;
         numScalarsPerTexel = 1;
        break;
      case TextureData::UCHAR4:
        desc         = cudaCreateChannelDesc<uchar4>();
        sizeOfScalar = 1;
        readMode     = cudaReadModeNormalizedFloat;
        numScalarsPerTexel = 4;
        break;
      case TextureData::USHORT:
        desc         = cudaCreateChannelDesc<uint16_t>();
        sizeOfScalar = 2;
        readMode     = cudaReadModeNormalizedFloat;
        numScalarsPerTexel = 1;
        break;
      default:
        BARNEY_NYI();
      };
      
      perDev.resize(dg->devices.size());
      for (auto dev : dg->devices) {
        auto &dd = perDev[dev->localID];
        SetActiveGPU forDuration(dev);

        unsigned int padded_x = (unsigned)dims.x;
        unsigned int padded_y = std::max(1u,(unsigned)dims.y);
        unsigned int padded_z = std::max(1u,(unsigned)dims.z);
        cudaExtent extent{padded_x,padded_y,padded_z};
        BARNEY_CUDA_CALL(Malloc3DArray(&dd.array,&desc,extent,0));
        cudaMemcpy3DParms copyParms;
        memset(&copyParms,0,sizeof(copyParms));
        copyParms.srcPtr
          = make_cudaPitchedPtr((void *)texels,
                                (size_t)padded_x*sizeOfScalar*numScalarsPerTexel,
                                (size_t)padded_x,
                                (size_t)padded_y);
        copyParms.dstArray = dd.array;
        copyParms.extent   = extent;
        copyParms.kind     = cudaMemcpyHostToDevice;
        BARNEY_CUDA_CALL(Memcpy3D(&copyParms));
      }
    }
    
    rtc::TextureData *
    BaseDevGroup::createTextureData(vec3i dims,
                                    rtc::TextureData::Format format,
                                    const void *texels) const
    {
      return new TextureData(this,dims,format,texels);
    }

    // inline cudaTextureReadMode toCUDA(rtc::Texture::ReadMode mode)
    // {
    //   return (mode == rtc::Texture::NORMALIZED_FLOAT)
    //     ? cudaReadModeNormalizedFloat
    //     : cudaReadModeElementType;
    // }
    
    inline cudaTextureFilterMode toCUDA(rtc::Texture::FilterMode mode)
    {
      return (mode == rtc::Texture::FILTER_MODE_POINT)
        ? cudaFilterModePoint
        : cudaFilterModeLinear;
    }
    
    inline cudaTextureAddressMode toCUDA(rtc::Texture::AddressMode mode)
    {
      switch (mode) {
      case rtc::Texture::MIRROR:
        return cudaAddressModeMirror;
      case rtc::Texture::CLAMP:
        return cudaAddressModeClamp;
      case rtc::Texture::WRAP:
        return cudaAddressModeWrap;
      case rtc::Texture::BORDER:
        return cudaAddressModeBorder;
      };
      // just to make teh compiler happy:
      return cudaAddressModeMirror;
    }

    Texture::Texture(const BaseDevGroup *dg,
                     TextureData *data,
                     rtc::Texture::FilterMode filterMode,
                     rtc::Texture::AddressMode addressModes[3],
                     const vec4f borderColor,
                     bool normalizedCoords,
                     rtc::Texture::ColorSpace colorSpace)
      : rtc::Texture(data)
    {
      perDev.resize(dg->devices.size());
      for (auto dev : dg->devices) {
        auto &dd = perDev[dev->localID];
        auto &dataDD
          = data->perDev[dev->localID];
        cudaResourceDesc resourceDesc;
        memset(&resourceDesc,0,sizeof(resourceDesc));
        resourceDesc.resType         = cudaResourceTypeArray;
        resourceDesc.res.array.array = dataDD.array;
        
        cudaTextureDesc textureDesc;
        memset(&textureDesc,0,sizeof(textureDesc));
        textureDesc.addressMode[0] = toCUDA(addressModes[0]);
        textureDesc.addressMode[1] = toCUDA(addressModes[1]);
        textureDesc.addressMode[2] = toCUDA(addressModes[2]);
        textureDesc.filterMode     = toCUDA(filterMode);
        textureDesc.readMode       = data->readMode;
        
        textureDesc.normalizedCoords = normalizedCoords;
        
        BARNEY_CUDA_CALL(CreateTextureObject(&dd,
                                             &resourceDesc,
                                             &textureDesc,0));
      }
    }
    
    
    rtc::device::TextureObject
    Texture::getDD(const rtc::Device *device) const
    {
      return (const rtc::device::TextureObject&)perDev[device->localID];
    }
    
    rtc::Texture *
    BaseDevGroup::createTexture(rtc::TextureData *data,
                                rtc::Texture::FilterMode filterMode,
                                rtc::Texture::AddressMode addressModes[3],
                                const vec4f borderColor,
                                bool normalizedCoords,
                                rtc::Texture::ColorSpace colorSpace) const
    {
      return new Texture(this,(TextureData*)data,
                         filterMode,addressModes,
                         borderColor,normalizedCoords,colorSpace);
    }

  }
}

