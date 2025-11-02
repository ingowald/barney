// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/embree/Device.h"

namespace rtc {
  namespace embree {
    
    struct TextureSampler;
    struct Texture;
    struct TextureData;

    /*! abstract interface for a texture sampler; derived classes will
        inmplement this based on data provided */
    struct TextureSampler {
      TextureSampler(TextureData *data,
                     const rtc::TextureDesc &desc)
        : data(data), desc(desc)
      {}
      
      virtual vec4f tex1D(float x) = 0;
      virtual vec4f tex2D(vec2f tc) = 0;
      virtual vec4f tex3D(vec3f tc) = 0;
      
      TextureData     *const data;
      rtc::TextureDesc const desc;
    };

    struct TextureData 
    {
      TextureData(Device *device,
                  vec3i dims,
                  rtc::DataType format,
                  const void *texels);
      Texture *createTexture(const rtc::TextureDesc &desc);
      
      size_t sizeOfScalar;
      size_t numScalarsPerTexel;
      const vec3i dims;
      const DataType format;
      std::vector<uint8_t> data;
      Device *const device;
    };


    struct Texture
    {
      Texture(TextureData *const data,
              const rtc::TextureDesc &desc);
      rtc::TextureObject getDD() const;

      TextureSampler *sampler = 0;
    };
      
  }
}
