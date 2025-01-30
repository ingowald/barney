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

namespace barney {
  namespace optix {

#if OPTIX_VERSION >= 80000
    struct Denoiser : public rtc::Denoiser {
      Denoiser(Device *device);
      virtual ~Denoiser();
      void resize(vec2i dims) override;
      void run(vec4f *out_rgba,
               vec4f *in_rgba,
               vec3f *in_normal,
               float blendFactor) override;
      Device *const device;

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

