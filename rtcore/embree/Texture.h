// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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
