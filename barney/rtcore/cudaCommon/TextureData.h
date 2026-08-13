// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/cudaCommon/cuda-common.h"

namespace BARNEY_NS {
  namespace rtc {

    struct CudaDeviceBase;
    struct Texture;
    
    struct TextureData
    {
      TextureData(CudaDeviceBase *device,
                  vec3i dims,
                  rtc::DataType format,
                  const void *texels);
      virtual ~TextureData();
      
      Texture *
      createTexture(const rtc::TextureDesc &desc);
      
      cudaArray_t           array;
      cudaTextureReadMode   readMode;
      const vec3i           dims;
      const DataType        format;
      CudaDeviceBase *const device;
    };
    
  }
}
