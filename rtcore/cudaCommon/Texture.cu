// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

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
    
  }
}
