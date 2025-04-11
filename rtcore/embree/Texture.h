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

namespace rtc {
  namespace embree {
    
    struct TextureSampler;
    struct Texture;
    struct TextureData;

    /*! abstract interface for a texture sampler; derived classes will
        inmplement this based on data provided */
    struct TextureSampler {
      TextureSampler(TextureData *data,
                     const rtc::TextureDesc &desc)
        : data(data), desc(desc)
      {}
      
      virtual vec4f tex1D(float x) = 0;
      virtual vec4f tex2D(vec2f tc) = 0;
      virtual vec4f tex3D(vec3f tc) = 0;
      
      TextureData     *const data;
      rtc::TextureDesc const desc;
    };

    struct TextureData 
    {
      TextureData(Device *device,
                  vec3i dims,
                  rtc::DataType format,
                  const void *texels);
      Texture *createTexture(const rtc::TextureDesc &desc);
      
      size_t sizeOfScalar;
      size_t numScalarsPerTexel;
      const vec3i dims;
      const DataType format;
      std::vector<uint8_t> data;
      Device *const device;
    };


    struct Texture
    {
      Texture(TextureData *const data,
              const rtc::TextureDesc &desc);
      rtc::TextureObject getDD() const;

      TextureSampler *sampler = 0;
    };
      
  }
}
