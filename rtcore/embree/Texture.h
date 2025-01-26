#pragma once

#include "rtcore/embree/Device.h"

namespace barney {
  namespace embree {
    
    struct TextureSampler;
    struct Texture;
    struct TextureData;
    
    struct TextureData : public rtc::TextureData
    {
      TextureData(Device *device,
                  vec3i dims,
                  rtc::DataType format,
                  const void *texels);
      rtc::Texture *createTexture(const rtc::TextureDesc &desc) override;
      
      size_t sizeOfScalar;
      size_t numScalarsPerTexel;
      std::vector<uint8_t> data;
      Device *const device;
    };


    struct Texture : public rtc::Texture
    {
      Texture(TextureData *const data,
              const rtc::TextureDesc &desc);
      rtc::device::TextureObject getDD() const override;

      TextureSampler *sampler = 0;
    };
      
  }
}
