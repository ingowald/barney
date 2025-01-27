// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

#include "barney/common/Texture.h"
#include "barney/Context.h"
#include "barney/ModelSlot.h"
#include "barney/DeviceGroup.h"

namespace barney {

  TextureData::PLD *TextureData::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank >= 0);
    assert(device->contextRank < perLogical.size());
    return &perLogical[device->contextRank];
  }
  Texture::PLD *Texture::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank >= 0);
    assert(device->contextRank < perLogical.size());
    return &perLogical[device->contextRank];
  }
  Texture3D::PLD *Texture3D::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank >= 0);
    assert(device->contextRank < perLogical.size());
    return &perLogical[device->contextRank];
  }
  
  rtc::Texture::ColorSpace toRTC(BNTextureColorSpace mode)
  { return 
      (mode == BN_COLOR_SPACE_LINEAR)
      ? rtc::Texture::COLOR_SPACE_LINEAR
      : rtc::Texture::COLOR_SPACE_SRGB;
  }
  
  rtc::Texture::FilterMode toRTC(BNTextureFilterMode mode)
  { return 
      (mode == BN_TEXTURE_NEAREST)
      ? rtc::Texture::FILTER_MODE_POINT
      : rtc::Texture::FILTER_MODE_LINEAR;
  }
  
  rtc::Texture::AddressMode toRTC(BNTextureAddressMode mode)
  {
    switch(mode) {
    case BN_TEXTURE_WRAP: return rtc::Texture::WRAP;
    case BN_TEXTURE_MIRROR: return rtc::Texture::MIRROR;
    case BN_TEXTURE_CLAMP: return rtc::Texture::CLAMP;
    case BN_TEXTURE_BORDER: return rtc::Texture::BORDER;
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
      
    case BN_FLOAT2:
    case BN_INT2:
      return 2;
      
    case BN_FLOAT3:
    case BN_INT3:
      return 3;
      
    case BN_FLOAT4:
    case BN_INT4:
    case BN_UFIXED8_RGBA:
    case BN_FLOAT4_RGBA:
      return 4;
    default:
      BARNEY_NYI();
    };
  }
  // in common/Data.cpp
  rtc::DataType toRTC(BNDataType format);

  Texture::Texture(Context *context, 
                   const DevGroup::SP &devices,
                   TextureData::SP data,
                   BNTextureFilterMode  filterMode,
                   BNTextureAddressMode addressMode_x,
                   BNTextureAddressMode addressMode_y,
                   BNTextureColorSpace  colorSpace)
    : SlottedObject(context,devices),
      data(data)
  {
    perLogical.resize(devices->numLogical);
    rtc::TextureDesc desc;
    desc.filterMode     = toRTC(filterMode);
    desc.addressMode[0] = toRTC(addressMode_x);
    desc.addressMode[1] = toRTC(addressMode_y);
    desc.colorSpace     = toRTC(colorSpace);
    for (auto device : *devices) {
      auto pld = getPLD(device);
      assert(pld);
      pld->rtcTexture
        = data->getPLD(device)->rtc->createTexture(desc);
    }
  }
  
  Texture3D::Texture3D(Context *context,
                       const DevGroup::SP &devices,
                       TextureData::SP data,
                       BNTextureFilterMode  filterMode,
                       BNTextureAddressMode addressMode)
    : SlottedObject(context,devices),
      data(data)
  {
    perLogical.resize(devices->numLogical);
    rtc::TextureDesc desc;
    desc.addressMode[0] = toRTC(addressMode);
    desc.addressMode[1] = toRTC(addressMode);
    desc.addressMode[2] = toRTC(addressMode);
    for (auto device : *devices) {
      auto pld = getPLD(device);
      desc.filterMode = toRTC(filterMode);
      pld->rtcTextureNN 
        = data->getPLD(device)->rtc->createTexture(desc);
      desc.filterMode = rtc::Texture::FILTER_MODE_POINT;
      pld->rtcTexture
        = data->getPLD(device)->rtc->createTexture(desc);
    }
  }

  Texture3D::DD Texture3D::getDD(Device *device)
  {
    Texture3D::DD dd;
    dd.texObj   = getPLD(device)->rtcTexture->getDD();
    dd.texObjNN = getPLD(device)->rtcTextureNN->getDD();
    return dd;
  }
  
  TextureData::TextureData(Context *context,
                           const DevGroup::SP &devices,
                           BNDataType texelFormat,
                           vec3i size,
                           const void *texels)
    : SlottedObject(context,devices),
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

  rtc::device::TextureObject
  Texture::getTextureObject(Device *device) 
  {
    auto pld = getPLD(device);
    assert(pld);
    assert(pld->rtcTexture);
    return pld->rtcTexture->getDD();
  }
    
  
}
