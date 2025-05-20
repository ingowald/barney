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
    case BN_TEXTURE_WRAP: return rtc::WRAP;
    case BN_TEXTURE_MIRROR: return rtc::MIRROR;
    case BN_TEXTURE_CLAMP: return rtc::CLAMP;
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
      
    case BN_FLOAT2:
    case BN_INT2:
      return 2;
      
    case BN_FLOAT3:
    case BN_INT3:
      return 3;
      
    case BN_FLOAT4:
    case BN_INT4:
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

#if 0
  Texture3D::PLD *Texture3D::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank >= 0);
    assert(device->contextRank < perLogical.size());
    return &perLogical[device->contextRank];
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
    // we do 3d texturing in non-normalized colors so integer coordinate is cell ID
    desc.normalizedCoords = false;
    for (auto device : *devices) {
      auto pld = getPLD(device);
      desc.filterMode = rtc::FILTER_MODE_POINT;
      pld->rtcTextureNN 
        = data->getPLD(device)->rtc->createTexture(desc);
      
      desc.filterMode = toRTC(filterMode);
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
#endif
  
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
