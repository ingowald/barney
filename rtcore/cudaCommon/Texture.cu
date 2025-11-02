// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "rtcore/cudaCommon/Device.h"
#include "rtcore/cudaCommon/TextureData.h"
#include "rtcore/cudaCommon/Texture.h"

namespace rtc {
  namespace cuda_common {

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
      // just to make the compiler happy:
      return cudaAddressModeMirror;
    }

    Texture::Texture(TextureData *data,
                     const TextureDesc &desc)
      : data(data),
        device(data->device)
    {
      SetActiveGPU forDuration(device);
        
      cudaResourceDesc resourceDesc;
      memset(&resourceDesc,0,sizeof(resourceDesc));
      resourceDesc.resType         = cudaResourceTypeArray;
      resourceDesc.res.array.array = data->array;
      
      cudaTextureDesc textureDesc;
      memset(&textureDesc,0,sizeof(textureDesc));
      textureDesc.addressMode[0]   = toCUDA(desc.addressMode[0]);
      textureDesc.addressMode[1]   = toCUDA(desc.addressMode[1]);
      textureDesc.addressMode[2]   = toCUDA(desc.addressMode[2]);
      textureDesc.filterMode       = toCUDA(desc.filterMode);
      textureDesc.readMode         = data->readMode;
      textureDesc.borderColor[0]   = desc.borderColor.x;
      textureDesc.borderColor[1]   = desc.borderColor.y;
      textureDesc.borderColor[2]   = desc.borderColor.z;
      textureDesc.borderColor[3]   = desc.borderColor.w;
      textureDesc.normalizedCoords = desc.normalizedCoords;
      
      BARNEY_CUDA_CALL(CreateTextureObject(&textureObject,
                                           &resourceDesc,
                                           &textureDesc,0));
    }

    Texture::~Texture()
    {
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_CALL_NOTHROW(DestroyTextureObject(textureObject));
      textureObject = 0;
    }
    
  }
}
