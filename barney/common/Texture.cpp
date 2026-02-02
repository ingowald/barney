// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/common/Texture.h"
#include "barney/Context.h"
#include "barney/ModelSlot.h"
#include "barney/DeviceGroup.h"

namespace BARNEY_NS {

  TextureData::PLD *TextureData::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank() >= 0);
    assert(device->contextRank() < perLogical.size());
    return &perLogical[device->contextRank()];
  }
  
  Texture::PLD *Texture::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank() >= 0);
    assert(device->contextRank() < perLogical.size());
    return &perLogical[device->contextRank()];
  }

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
      return 1;
      
    case BN_FLOAT32_VEC2:
    case BN_INT32_VEC2:
    case BN_UINT32_VEC2:
      return 2;
      
    case BN_FLOAT32_VEC3:
    case BN_INT32_VEC3:
    case BN_UINT32_VEC3:
      return 3;
      
    case BN_FLOAT32_VEC4:
    case BN_INT32_VEC4:
    case BN_UINT32_VEC4:
    case BN_UFIXED8_RGBA:
      return 4;
    default:
      BARNEY_NYI();
    };
  }
  // in common/Data.cpp
  rtc::DataType toRTC(BNDataType format);

  Texture::Texture(Context *context, 
                   TextureData::SP data,
                   // const vec4f          borderColor,
                   BNTextureFilterMode  filterMode,
                   BNTextureAddressMode addressModes[],
                   BNTextureColorSpace  colorSpace)
    : barney_api::Texture(context),
      devices(data->devices),
      data(data)
  {
    perLogical.resize(devices->numLogical);
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
    for (auto device : *devices) {
      auto pld = getPLD(device);
      assert(pld);
      pld->rtcTexture
        = data->getPLD(device)->rtc->createTexture(desc);
    }
  }

  TextureData::TextureData(Context *context,
                           const DevGroup::SP &devices,
                           BNDataType texelFormat,
                           vec3i size,
                           const void *texels)
    : barney_api::TextureData(context),
      devices(devices),
      dims(size),
      numChannels(numChannelsOf(texelFormat)),
      texelFormat(texelFormat)
  {
    perLogical.resize(devices->numLogical);
    rtc::DataType format = toRTC(texelFormat);
    for (auto device : *devices) {
      auto pld = getPLD(device);
      pld->rtc
        = device->rtc->createTextureData(size,format,texels);
      assert(pld->rtc);
    }
  }

  TextureData::~TextureData()
  {
    for (auto device : *devices)
      device->rtc->freeTextureData(getPLD(device)->rtc);
  }

  rtc::TextureObject
  Texture::getTextureObject(Device *device) 
  {
    auto pld = getPLD(device);
    assert(pld);
    assert(pld->rtcTexture);
    return pld->rtcTexture->getDD();
  }
  
}
