// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/cudaCommon/cuda-common.h"

namespace rtc {
  namespace cuda_common {
    
    struct Texture
    {
      Texture(TextureData *data,
              const TextureDesc &desc);
      virtual ~Texture();
      
      TextureObject getDD() const
      { return (const TextureObject&)textureObject; }

      Device             *const device;
      TextureData        *const data;
      cudaTextureObject_t textureObject;
    };
    
  }
}
