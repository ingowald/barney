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

#include "rtcore/optix/Device.h"

namespace rtc {
  namespace optix {

    /*! abstract interface to a denoiser. implementation(s) depend of
        which optix version and/or oidn are available */
    struct Denoiser {
      Denoiser(Device* device) : device(device) {}
      virtual ~Denoiser() = default;
      virtual void resize(vec2i dims) = 0;
      virtual void run(float blendFactor) = 0;
      vec4f *out_rgba  = 0;
      vec4f *in_rgba   = 0;
      vec3f *in_normal = 0;
      Device* const device;
    };

#if OPTIX_VERSION >= 80000
    /*! denoising using optix 8 built-in denoiser. only available for
        optix 8 or newer */
    struct Optix8Denoiser : public Denoiser {
      Optix8Denoiser(Device *device);
      virtual ~Optix8Denoiser();
      void resize(vec2i dims) override;
      void run(float blendFactor) override;
      
      vec2i                numPixels;
      OptixDenoiser        denoiser = {};
      OptixDenoiserOptions denoiserOptions;
      void                *denoiserScratch = 0;
      void                *denoiserState   = 0;
      OptixDenoiserSizes   denoiserSizes;
    };
#endif
    
  }
}

