// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/common/Texture.h"
#include "native/Context.h"
#include "native/ModelSlot.h"
#include "native/DeviceGroup.h"

namespace BARNEY_NS {
  namespace native {
    
    std::string TextureData::toString() const 
    { return "TextureData{}"; }
    
    std::string Texture::toString() const 
    { return "Texture{}"; }
    
    // TextureData::PLD *TextureData::getPLD(Device *device) 
    // {
    //   assert(device);
    //   assert(device->contextRank() >= 0);
    //   assert(device->contextRank() < perLogical.size());
    //   return &perLogical[device->contextRank()];
    // }
  
    // Texture::PLD *Texture::getPLD(Device *device) 
    // {
    //   assert(device);
    //   assert(device->contextRank() >= 0);
    //   assert(device->contextRank() < perLogical.size());
    //   return &perLogical[device->contextRank()];
    // }

    rtc::ColorSpace toRTC(BNTextureColorSpace mode)
    { return 
        (mode == BN_COLOR_SPACE_LINEAR)
        ? rtc::COLOR_SPACE_LINEAR
        : rtc::COLOR_SPACE_SRGB;
    }
  
    rtc::FilterMode toRTC(BNTextureFilterMode mode)
    { return 
        (mode == BN_TEXTURE_NEAREST)
        ? rtc::FILTER_MODE_POINT
        : rtc::FILTER_MODE_LINEAR;
    }
  
    rtc::AddressMode toRTC(BNTextureAddressMode mode)
    {
      switch(mode) {
      case BN_TEXTURE_WRAP:   return rtc::WRAP;
      case BN_TEXTURE_MIRROR: return rtc::MIRROR;
      case BN_TEXTURE_CLAMP:  return rtc::CLAMP;
      case BN_TEXTURE_BORDER: return rtc::BORDER;
      default:
        BARNEY_INVALID_VALUE();
      }
    }

    int numChannelsOf(BNDataType dataType)
    {
      switch(dataType) {
      case BN_FLOAT:
      case BN_UFIXED8:
      case BN_UFIXED16:
      case BN_FLOAT16:
        return 1;
      
      case BN_FLOAT32_VEC2:
      case BN_INT32_VEC2:
      case BN_UINT32_VEC2:
      case BN_UFIXED8_VEC2:
      case BN_UFIXED16_VEC2:
      case BN_FLOAT16_VEC2:
        return 2;
      
      case BN_FLOAT32_VEC3:
      case BN_INT32_VEC3:
      case BN_UINT32_VEC3:
      case BN_FLOAT16_VEC3:
        return 3;
      
      case BN_FLOAT32_VEC4:
      case BN_INT32_VEC4:
      case BN_UINT32_VEC4:
      case BN_UFIXED8_RGBA:
      case BN_UFIXED8_RGBA_SRGB:
      case BN_FLOAT16_VEC4:
        return 4;
      default:
        BARNEY_NYI();
      };
    }
    // in common/Data.cpp
    rtc::DataType toRTC(BNDataType format);

    Texture::Texture(TextureData::SP data,
                     // const vec4f          borderColor,
                     BNTextureFilterMode  filterMode,
                     BNTextureAddressMode addressModes[],
                     BNTextureColorSpace  colorSpace)
      : Object(data->device),
        data(data)
    {
      rtc::TextureDesc desc;
      desc.filterMode     = toRTC(filterMode);
      // desc.borderColor    = borderColor;
    
      if (data->dims[2] > 0)
        desc.normalizedCoords = false;

      for (int i=0;i<3;i++)
        if (data->dims[i] > 0)
          desc.addressMode[i] = toRTC(addressModes[i]);
      // desc.addressMode[1] = toRTC(addressMode_y);
      desc.colorSpace     = toRTC(colorSpace);
      rtcTexture
        = data->rtc->createTexture(desc);
    }

    TextureData::TextureData(Device *device,
                             BNDataType texelFormat,
                             vec3i size,
                             const void *texels)
      : Object(device),
        dims(size),
        numChannels(numChannelsOf(texelFormat)),
        texelFormat(texelFormat)
    {
      rtc::DataType format = toRTC(texelFormat);
      // for (auto device : *devices) {
      //   auto pld = getPLD(device);
      this->rtc
        = device->rtc->createTextureData(size,format,texels);
    }

    Texture::~Texture()
    {
        SetActiveGPU forDuration(device);
        if (rtcTexture) {
          device->rtc->freeTexture(rtcTexture);
          rtcTexture = 0;
      }
    }

    TextureData::~TextureData()
    {
        device->rtc->freeTextureData(this->rtc);
    }

    rtc::TextureObject
    Texture::getTextureObject() 
    {
      // auto pld = getPLD(device);
      // assert(pld);
      // assert(pld->rtcTexture);
      // return pld->rtcTexture->getDD();
      return rtcTexture->getDD();
    }

  }
}
