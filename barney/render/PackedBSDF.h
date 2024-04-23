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

#include "barney/render/packedBSDFs/VisRTX.h"
#include "barney/render/packedBSDFs/Glass.h"
#include "barney/render/packedBSDFs/Phase.h"

namespace barney {
  namespace render {

    namespace packedBSDF {
      struct Invalid { };
    }
    
    struct PackedBSDF {
      typedef enum { INVALID=0, NONE=INVALID,
        /* phase function */
        TYPE_Phase,
        TYPE_VisRTX
      } Type;
      struct Data {
        union {
          packedBSDF::Phase  phase;
          packedBSDF::VisRTX visRTX;
          packedBSDF::Glass  glass;
        };
      } data;

      Type type;

      inline __device__ PackedBSDF();
      inline __device__ PackedBSDF(const packedBSDF::Invalid &invalid)
      { type = INVALID; }
      inline __device__ PackedBSDF(const packedBSDF::Phase  &phase)
      { type = TYPE_Phase; data.phase = phase; }
      inline __device__ PackedBSDF(const packedBSDF::VisRTX &visRTX);
      
      inline __device__
      EvalRes eval(render::DG dg, vec3f w_i, bool dbg=false) const;
      
      inline __device__
      float getOpacity(render::DG dg, bool dbg=false) const;
    };


    inline __device__
    EvalRes PackedBSDF::eval(render::DG dg, vec3f w_i, bool dbg) const
    {
      if (type == TYPE_Phase)
        return data.phase.eval(dg,w_i,dbg);
      if (type == TYPE_VisRTX)
        return data.visRTX.eval(dg,w_i,dbg);
      return EvalRes();
    }
    
    inline __device__
    PackedBSDF::PackedBSDF(const packedBSDF::VisRTX &visRTX)
    { type = TYPE_VisRTX; data.visRTX = visRTX; }
    
    inline __device__
    float PackedBSDF::getOpacity(render::DG dg, bool dbg) const
    {
      return 0.f;
    }
    
  }
}
