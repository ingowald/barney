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

#pragma once

#include "barney/Object.h"
#include "rtcore/Frontend.h"

namespace BARNEY_NS {

  rtc::ColorSpace toRTC(BNTextureColorSpace mode);
  rtc::FilterMode toRTC(BNTextureFilterMode mode);
  rtc::AddressMode toRTC(BNTextureAddressMode mode);
  
  struct Device;
  struct Context;
  
  struct TextureData : public barney_api::TextureData {
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


  struct Texture : public barney_api::Texture {
    typedef std::shared_ptr<Texture> SP;

    Texture(Context *context,
            TextureData::SP data, 
            BNTextureFilterMode  filterMode,
            BNTextureAddressMode addressModes[],
            BNTextureColorSpace  colorSpace);
    virtual ~Texture() = default;

    rtc::device::TextureObject getTextureObject(Device *device);
    rtc::device::TextureObject getDD(Device *device)
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

  // struct Texture3D : public SlottedObject {
  //   typedef std::shared_ptr<Texture3D> SP;

  //   struct DD {
  //     rtc::device::TextureObject texObj;
  //     rtc::device::TextureObject texObjNN;
  //   };

  //   Texture3D(Context *context,
  //             const DevGroup::SP &devices,
  //             TextureData::SP data,
  //             BNTextureFilterMode  filterMode,
  //             BNTextureAddressMode addressMode);
  //   virtual ~Texture3D() = default;
    
  //   struct PLD {
  //     rtc::Texture *rtcTexture = 0;
  //     rtc::Texture *rtcTextureNN = 0;
  //   };
    
  //   PLD *getPLD(Device *device);
  //   std::vector<PLD> perLogical;
    
  //   /*! pretty-printer for printf-debugging */
  //   std::string toString() const override
  //   { return "Texture3D{}"; }

  //   DD getDD(Device *device);
  // private:
  //   TextureData::SP data;
  // };

}
