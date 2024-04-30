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

#include "barney/render/device/DG.h"

namespace barney {
  namespace render {
    namespace packedBSDF {
      
      struct Glass {
        inline __device__ vec3f getAlbedo(bool dbg) const;
        inline __device__ float getOpacity(render::DG dg, bool dbg=false) const;
        inline __device__ EvalRes eval(DG dg, vec3f wi, bool dbg) const;

        float  ior;
        float3 attenuation;
      };
      
    }
  }
}

