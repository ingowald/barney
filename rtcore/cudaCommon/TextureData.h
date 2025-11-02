// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/cudaCommon/cuda-common.h"

namespace rtc {
  namespace cuda_common {
    
    struct Device;
    struct Texture;
    
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
    
  }
}
