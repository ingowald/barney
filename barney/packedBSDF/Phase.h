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

#include "barney/render/DG.h"

namespace BARNEY_NS {
  namespace render {
    namespace packedBSDF {

      /*! implements a homogenous phase function that scatters equally
          in all directions, with given average reflectance and
          color */
      struct Phase {
        inline Phase() = default;
        inline __both__ Phase(vec3f color, float avg_reflectance=.7f);

        inline __both__
        float pdf(DG dg, vec3f wi, bool dbg) const;
        
        inline __both__
        EvalRes eval(DG dg, vec3f wi, bool dbg) const;
        
        inline __both__
        void scatter(ScatterResult &scatter,
                     const render::DG &dg,
                     Random &random,
                     bool dbg) const;
        
        rtc::float3 albedo;
      };

      inline __both__
      Phase::Phase(vec3f color, float avg_reflectance)
      {
        (vec3f&)this->albedo = avg_reflectance * color;
      }
      
      inline __both__
      float Phase::pdf(DG dg, vec3f wi, bool dbg) const
      { return ONE_OVER_FOUR_PI; }
        
      inline __both__
      EvalRes Phase::eval(DG dg, vec3f wi, bool dbg) const
      {
        float density = ONE_OVER_FOUR_PI;
        return EvalRes(//density*
                       (const vec3f&)albedo,density);
      }

      /*! simple omnidirectional phase function - scatter into any
        random direction */
      inline __both__
      void Phase::scatter(ScatterResult &scatter,
                          const render::DG &dg,
                          Random &random,
                          bool dbg) const
      {
        // see global illumination compendium, page 19
        float r1 = random();
        float r2 = random(); 
        // float phi = two_pi*r1;
        // float theta = acosf(1.f-2.f*r2);
        float x = cosf(TWO_PI*r1)*sqrtf(r2*(1.f-r2));
        float y = sinf(TWO_PI*r1)*sqrtf(r2*(1.f-r2));
        float z = (1.f-2.f*r2);
        float density = ONE_OVER_FOUR_PI;
        scatter.pdf = density;
        scatter.f_r = (const vec3f&)albedo;// * density;
        scatter.dir = vec3f(x,y,z);
        scatter.type = ScatterResult::VOLUME;
      }
      
    }
  }
}

