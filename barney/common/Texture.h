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
#include <barney.h>

namespace barney {

  rtc::Texture::ColorSpace toRTC(BNTextureColorSpace mode);
  rtc::Texture::FilterMode toRTC(BNTextureFilterMode mode);
  rtc::Texture::AddressMode toRTC(BNTextureAddressMode mode);
  
  struct Device;
  
  struct Texture : public SlottedObject {
    typedef std::shared_ptr<Texture> SP;

    Texture(Context *context, int slot,
            BNDataType texelFormat,
            vec2i size,
            const void *texels,
            BNTextureFilterMode  filterMode,
            BNTextureAddressMode addressMode_x,
            BNTextureAddressMode addressMode_y,
            BNTextureColorSpace  colorSpace);
    virtual ~Texture() = default;

    // cudaTextureObject_t getTextureObject(const Device *device) const;
    rtc::device::TextureObject getTextureObject(const Device *device) const;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Texture{}"; }

    rtc::TextureData *rtcTextureData = 0;
    rtc::Texture *rtcTexture = 0;
    // OWLTexture owlTexture = 0;
  };

  struct Texture3D : public SlottedObject {
    typedef std::shared_ptr<Texture3D> SP;

    struct DD {
      // cudaArray_t           voxelArray = 0;
      // cudaTextureObject_t   texObj;
      // cudaTextureObject_t   texObjNN;
      rtc::device::TextureObject texObj;
      rtc::device::TextureObject texObjNN;
    };
    
    Texture3D(Context *context, int slot,
              BNDataType texelFormat,
              vec3i size,
              const void *texels,
              BNTextureFilterMode  filterMode,
              BNTextureAddressMode addressMode);
    virtual ~Texture3D() = default;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Texture3D{}"; }

    DD &getDD(const std::shared_ptr<Device> &device);
  private:
    rtc::TextureData *rtcTextureData = 0;
    rtc::Texture *rtcTexture = 0;
    rtc::Texture *rtcTextureNN = 0;
    /*! one tex3d per device */
    // std::vector<DD> tex3Ds;
  };

  struct TextureData : public SlottedObject {
    typedef std::shared_ptr<TextureData> SP;

    // struct DD {
    //   cudaArray_t array = 0;
    // };
    
    /*! one cudaArray per device */
    TextureData(Context *context, int slot,
                BNDataType texelFormat,
                vec3i size,
                const void *texels);
    virtual ~TextureData();

    // DD &getDD(const std::shared_ptr<Device> &device);
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "TextureData{}"; }

    rtc::TextureData *rtcTextureData = 0;
    // std::vector<DD> onDev;
    vec3i           dims;
    BNDataType      texelFormat;
  };
}
