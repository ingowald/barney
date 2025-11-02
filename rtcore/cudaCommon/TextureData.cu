// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "rtcore/cudaCommon/Device.h"
#include "rtcore/cudaCommon/TextureData.h"

namespace rtc {
  namespace cuda_common {

    TextureData::TextureData(Device *device,
                             vec3i dims,
                             rtc::DataType format,
                             const void *texels)
      : device(device), dims(dims), format(format)
    {
      SetActiveGPU forDuration(device);
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
      case rtc::FLOAT3:
        throw std::runtime_error("float3 textures not allowed in barney::rtc::cuda");
      default:
        PRINT(std::to_string((int)format));
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
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_CALL_NOTHROW(FreeArray(array));
      array = 0;
    }
    
  }
}
