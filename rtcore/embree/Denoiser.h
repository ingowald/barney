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
#if BARNEY_OIDN_CPU
    struct Denoiser : public rtc::Denoiser
    {
      Denoiser(Device *device);
      void resize(vec2i size) override;
      void run(vec4f *out_rgba,
               vec3f *in_rgb,
               float *in_alpha,
               vec3f *in_normal) override;
      vec2i size { 0,0 }
    };
#endif
  }
}
