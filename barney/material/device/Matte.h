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

#include "barney/material/device/DG.h"
#include "barney/material/device/BSDF.h"
#include "barney/material/bsdfs/Lambert.h"

namespace barney {
  namespace render {

    typedef enum { IMAGE1D=0, TRANSFORM, NO_SAMPLER } SamplerType;

    struct Matte {
      struct HitBSDF {
        inline __device__
        vec3f getAlbedo(bool dbg=false) const { return (vec3f)lambert.albedo; }
        
        inline __device__
        EvalRes eval(render::DG dg, vec3f wi, bool dbg=false) const
        {
          return lambert.eval(dg,wi,dbg);
        }
        
        Lambert lambert;

        enum { bsdfType = Minneart::bsdfType | Lambert::bsdfType };
      };
      struct DD {
        inline __device__
        void make(HitBSDF &multi, vec3f geometryColor, bool dbg) const
        {
          multi.lambert.init(!isnan(geometryColor.x)
                             ? geometryColor
                             : reflectance,
                             dbg);
        }
        vec3f reflectance;
        SamplerType samplerType;
        //union { // not POD :( (TODO?!)
        struct {
          struct {
            int inAttribute;
            mat4f inTransform;
            vec4f inOffset;
            mat4f outTransform;
            vec4f outOffset;
            const vec4f *image;
            int imageSize;
          } image1D;
          struct {
            int inAttribute;
            mat4f outTransform;
            vec4f outOffset;
          } transform;
        } sampler;
      };
    };
    
  }
}