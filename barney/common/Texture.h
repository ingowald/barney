// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/Object.h"

namespace BARNEY_NS {
  namespace native {
    
  rtc::ColorSpace toRTC(BNTextureColorSpace mode);
  rtc::FilterMode toRTC(BNTextureFilterMode mode);
  rtc::AddressMode toRTC(BNTextureAddressMode mode);
  
  struct Device;
  struct Context;
  
  struct TextureData : public Object {
    typedef std::shared_ptr<TextureData> SP;

    struct PLD {
      rtc::TextureData *rtc = 0;
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;
   
    TextureData(Context *context,
                const DevGroup::SP &devices,
                BNDataType texelFormat,
                vec3i size,
                const void *texels);
    virtual ~TextureData();

    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "TextureData{}"; }

    int             numChannels;
    vec3i           dims;
    BNDataType      texelFormat;
    DevGroup::SP    const devices;
  };


  struct Texture : public Object {
    typedef std::shared_ptr<Texture> SP;

    Texture(Context *context,
            TextureData::SP data,
            BNTextureFilterMode  filterMode,
            BNTextureAddressMode addressModes[],
            BNTextureColorSpace  colorSpace);
    virtual ~Texture();

    rtc::TextureObject getTextureObject(Device *device);
    rtc::TextureObject getDD(Device *device)
    { return getTextureObject(device); }

    struct PLD {
      rtc::Texture *rtcTexture = 0;
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Texture{}"; }

    vec2i getDims() const { return {data->dims.x,data->dims.y}; }
    TextureData::SP data;
    DevGroup::SP    const devices;
  };

  }
}
