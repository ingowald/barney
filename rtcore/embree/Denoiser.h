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

#if BARNEY_OIDN_CPU
# include <OpenImageDenoise/oidn.h>
#endif

namespace rtc {
  namespace embree {

    struct Denoiser {
      Denoiser(Device *device) : rtc(device) {}
      virtual ~Denoiser() = default;
      virtual void resize(vec2i dims) = 0;
      virtual void run(vec4f* out_rgba,
                       vec4f* in_rgba,
                       vec3f* in_normal,
                       float blendFactor) = 0;
      Device *const rtc;
    };
    
#if BARNEY_OIDN_CPU
    /*! oidn-based CPU denoiser */
    struct DenoiserOIDN : public Denoiser
    {
      DenoiserOIDN(Device *device);
      virtual ~DenoiserOIDN();
      
      void resize(vec2i size) override;
      void run(// output
               vec4f *out_rgba,
               // input channels
               vec4f *in_rgba,
               vec3f *in_normal,
               float blendFactor) override;
      
      vec2i         numPixels { 0,0 };

      OIDNDevice oidnDevice = 0;
      OIDNFilter filter = 0;
    };
#endif
  }
}
