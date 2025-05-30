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

#include "rtcore/cudaCommon/cuda-common.h"

namespace rtc {
  namespace cuda_common {
    
    struct Texture
    {
      Texture(TextureData *data,
              const TextureDesc &desc);
      virtual ~Texture();
      
      TextureObject getDD() const
      { return (const TextureObject&)textureObject; }

      Device             *const device;
      TextureData        *const data;
      cudaTextureObject_t textureObject;
    };
    
  }
}
