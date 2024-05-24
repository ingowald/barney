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

    typedef enum { IMAGE1D=0, IMAGE2D, TRANSFORM, NO_SAMPLER } SamplerType;
    typedef enum { CLAMP=0, WRAP, MIRROR } WrapMode;

    struct Matte {
      struct HitBSDF {
        inline __device__
        vec3f getAlbedo(bool dbg=false) const
        { return Lambert(reflectance).getAlbedo(dbg); }
        
        inline __device__
        EvalRes eval(render::DG dg, vec3f wi, bool dbg=false) const
        {
          return Lambert(reflectance).eval(dg,wi,dbg);
        }
        
        vec3h reflectance;

        enum { bsdfType = Lambert::bsdfType };
      };
      struct DD {
        inline __device__
        void make(HitBSDF &multi, vec3f geometryColor, bool dbg) const;
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
            cudaTextureObject_t image;
          } image;
          struct {
            int inAttribute;
            mat4f outTransform;
            vec4f outOffset;
          } transform;
        } sampler;
      };
    };
    
    inline __device__
    void Matte::DD::make(HitBSDF &multi, vec3f geometryColor, bool dbg) const
    {
      multi.reflectance
        = !isnan(geometryColor.x)
        ? geometryColor
        : reflectance;
    }
    
  }
}
