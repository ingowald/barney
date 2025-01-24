#pragma once

#include "rtcore/embree/Device.h"

namespace barney {
  namespace embree {
    
    struct SWTexture;
    struct Texture;
    struct TextureData;
    
    struct TextureData : public rtc::TextureData
    {
      TextureData(Device *device,
                  vec3i dims,
                  rtc::DataType format,
                  const void *texels);
      rtc::Texture *createTexture(const rtc::TextureDesc &desc) override;
      
      Device *const device;
    };


    struct Texture : public rtc::Texture
    {
      Texture(TextureData *const data,
              const rtc::TextureDesc &desc);
      rtc::device::TextureObject getDD() const override
      { return (const rtc::device::TextureObject &)swTex; }

      SWTexture *swTex = 0;
    };
      
  }
}
