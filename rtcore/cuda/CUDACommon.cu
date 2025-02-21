#include "rtcore/cuda/CUDACommon.h"

namespace rtc {
  namespace cuda_common {

    // BaseBackend::BaseBackend()
    // {
    //   cudaFree(0);
    //   BARNEY_CUDA_CALL(GetDeviceCount(&numPhysicalDevices));
    // }
    
    int Device::setActive() const
    {
      int oldActive = 0;
      BARNEY_CUDA_CHECK(cudaGetDevice(&oldActive));
      BARNEY_CUDA_CHECK(cudaSetDevice(physicalID));
      return oldActive;
    }
    
    void Device::restoreActive(int oldActive) const
    {
      BARNEY_CUDA_CHECK(cudaSetDevice(oldActive));
    }
    
    void *Device::allocMem(size_t numBytes)
    {
      SetActiveGPU forDuration(this);
      void *ptr = 0;
      BARNEY_CUDA_CALL(Malloc((void **)&ptr,numBytes));
      assert(ptr);
      return ptr;
    }
    
    void *Device::allocHost(size_t numBytes) 
    {
      SetActiveGPU forDuration(this);
      void *ptr = 0;
      BARNEY_CUDA_CALL(MallocHost(&ptr,numBytes));
      return ptr;
    }
      
    void Device::freeHost(void *mem) 
    {
      SetActiveGPU forDuration(this);
      BARNEY_CUDA_CALL(FreeHost(mem));
    }
      
    void Device::freeMem(void *mem) 
    {
      SetActiveGPU forDuration(this);
      BARNEY_CUDA_CALL(Free(mem));
    }
      
    void Device::memsetAsync(void *mem,int value, size_t size) 
    {
      SetActiveGPU forDuration(this);
      BARNEY_CUDA_CALL(MemsetAsync(mem,value,size,stream));
    }
      

    void Device::copyAsync(void *dst, const void *src, size_t numBytes) 
    {
      SetActiveGPU forDuration(this);
      BARNEY_CUDA_CALL(MemcpyAsync(dst,src,numBytes,cudaMemcpyDefault,stream));
    }
      
    void Device::sync() 
    {
      SetActiveGPU forDuration(this);
      BARNEY_CUDA_CALL(StreamSynchronize(stream));
    }

    void Device::freeTextureData(TextureData *td)
    {
      if (td) delete td;
    }
    
    void Device::freeTexture(Texture *tex)
    {
      if (tex) delete tex;
    }
    
    TextureData::TextureData(Device *device,
                             vec3i dims,
                             rtc::DataType format,
                             const void *texels)
      : device(device), dims(dims), format(format)
    {
      cudaChannelFormatDesc desc;
      size_t sizeOfScalar;
      size_t numScalarsPerTexel;
      switch (format) {
      case rtc::FLOAT:
        desc         = cudaCreateChannelDesc<float>();
        sizeOfScalar = 4;
        readMode     = cudaReadModeElementType;
        numScalarsPerTexel = 1;
        break;
      case rtc::FLOAT4:
        desc         = cudaCreateChannelDesc<float4>();
        sizeOfScalar = 4;
        readMode     = cudaReadModeElementType;
        numScalarsPerTexel = 4;
        break;
      case rtc::UCHAR:
        desc         = cudaCreateChannelDesc<uint8_t>();
        sizeOfScalar = 1;
        readMode     = cudaReadModeNormalizedFloat;
        numScalarsPerTexel = 1;
        break;
      case rtc::UCHAR4:
        desc         = cudaCreateChannelDesc<uchar4>();
        sizeOfScalar = 1;
        readMode     = cudaReadModeNormalizedFloat;
        numScalarsPerTexel = 4;
        break;
      case rtc::USHORT:
        desc         = cudaCreateChannelDesc<uint16_t>();
        sizeOfScalar = 2;
        readMode     = cudaReadModeNormalizedFloat;
        numScalarsPerTexel = 1;
        break;
      default:
        assert(0);
      };

      if (dims.z != 0) {
        unsigned int padded_x = (unsigned)dims.x;
        unsigned int padded_y = std::max(1u,(unsigned)dims.y);
        unsigned int padded_z = std::max(1u,(unsigned)dims.z);
        cudaExtent extent{padded_x,padded_y,padded_z};
        BARNEY_CUDA_CALL(Malloc3DArray(&array,&desc,extent,0));
        cudaMemcpy3DParms copyParms;
        memset(&copyParms,0,sizeof(copyParms));
        copyParms.srcPtr
          = make_cudaPitchedPtr((void *)texels,
                                (size_t)padded_x*sizeOfScalar*numScalarsPerTexel,
                                (size_t)padded_x,
                                (size_t)padded_y);
        copyParms.dstArray = array;
        copyParms.extent   = extent;
        copyParms.kind     = cudaMemcpyHostToDevice;
        BARNEY_CUDA_CALL(Memcpy3D(&copyParms));
      } else if (dims.y != 0) {
        BARNEY_CUDA_CALL(MallocArray(&array,&desc,dims.x,dims.y,0));
        BARNEY_CUDA_CALL(Memcpy2DToArray(array,0,0,
                                         (void *)texels,
                                         (size_t)dims.x*sizeOfScalar*numScalarsPerTexel,
                                         (size_t)dims.x*sizeOfScalar*numScalarsPerTexel,
                                         (size_t)dims.y,
                                         cudaMemcpyHostToDevice));
      } else {
        assert(0);
      }
    }

    TextureData::~TextureData()
    {
      BARNEY_CUDA_CALL_NOTHROW(FreeArray(array));
      array = 0;
    }
    
    TextureData *
    Device::createTextureData(vec3i dims,
                              rtc::DataType format,
                              const void *texels) 
    {
      SetActiveGPU forDuration(this);
      return new TextureData(this,dims,format,texels);
    }

    inline cudaTextureFilterMode toCUDA(FilterMode mode)
    {
      return (mode == FILTER_MODE_POINT)
        ? cudaFilterModePoint
        : cudaFilterModeLinear;
    }
    
    inline cudaTextureAddressMode toCUDA(AddressMode mode)
    {
      switch (mode) {
      case MIRROR:
        return cudaAddressModeMirror;
      case CLAMP:
        return cudaAddressModeClamp;
      case WRAP:
        return cudaAddressModeWrap;
      case BORDER:
        return cudaAddressModeBorder;
      };
      // just to make teh compiler happy:
      return cudaAddressModeMirror;
    }

    Texture::Texture(TextureData *data,
                     const TextureDesc &desc)
      : data(data)//, desc(desc)
    {
      cudaResourceDesc resourceDesc;
      memset(&resourceDesc,0,sizeof(resourceDesc));
      resourceDesc.resType         = cudaResourceTypeArray;
      resourceDesc.res.array.array = data->array;
      
      cudaTextureDesc textureDesc;
      memset(&textureDesc,0,sizeof(textureDesc));
      textureDesc.addressMode[0] = toCUDA(desc.addressMode[0]);
      textureDesc.addressMode[1] = toCUDA(desc.addressMode[1]);
      textureDesc.addressMode[2] = toCUDA(desc.addressMode[2]);
      textureDesc.filterMode     = toCUDA(desc.filterMode);
      textureDesc.readMode       = data->readMode;
      textureDesc.borderColor[0] = desc.borderColor.x;
      textureDesc.borderColor[1] = desc.borderColor.y;
      textureDesc.borderColor[2] = desc.borderColor.z;
      textureDesc.borderColor[3] = desc.borderColor.w;
      textureDesc.normalizedCoords = desc.normalizedCoords;
      
      BARNEY_CUDA_CALL(CreateTextureObject(&textureObject,
                                           &resourceDesc,
                                           &textureDesc,0));
    }

    Texture::~Texture()
    {
      BARNEY_CUDA_CALL_NOTHROW(DestroyTextureObject(textureObject));
      textureObject = 0;
    }
    
    Texture *TextureData::createTexture(const TextureDesc &desc) 
    {
      SetActiveGPU forDuration(device);
      return new Texture(this,desc);
    }    
    
  }
}

