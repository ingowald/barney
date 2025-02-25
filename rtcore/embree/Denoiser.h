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

namespace barney {
  namespace embree {

    /*! oidn-based CPU denoiser */
    struct Denoiser
    {
      Denoiser(Device *device);
      virtual ~Denoiser();
      
      void resize(vec2i size);
      void run(// output
               vec4f *out_rgba,
               // input channels
               vec4f *in_rgba,
               vec3f *in_normal,
               float blendFactor);
      
      vec2i         numPixels { 0,0 };
      Device *const rtc;

      OIDNDevice oidnDevice = 0;
      OIDNFilter filter = 0;
    };
  }
}
#endif
