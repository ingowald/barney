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

#include "barney/render/DG.h"
#include "barney/packedBSDF/NVisii.h"
#include "barney/packedBSDF/Glass.h"
#include "barney/packedBSDF/Phase.h"
#include "barney/packedBSDF/Lambertian.h"

namespace BARNEY_NS {
  namespace render {

    namespace packedBSDF {
      struct Invalid { };
    }

    struct PackedBSDF {
      typedef enum {
        INVALID=0, NONE=INVALID,
        /* phase function */
        TYPE_Phase,
        TYPE_Glass,
        TYPE_Lambertian,
        TYPE_NVisii
      } Type;
      struct Data {
        union {
          packedBSDF::Phase      phase;
          packedBSDF::Lambertian lambertian;
          packedBSDF::Glass      glass;
          packedBSDF::NVisii     nvisii;
        };
      } data;

      Type type;

#if RTC_DEVICE_CODE
      inline __rtc_device PackedBSDF();
      inline __rtc_device PackedBSDF(Type type, Data data)
        : type(type), data(data) {}
      inline __rtc_device PackedBSDF(const packedBSDF::Invalid &invalid)
      { type = INVALID; }
      inline __rtc_device PackedBSDF(const packedBSDF::Phase  &phase)
      { type = TYPE_Phase; data.phase = phase; }
      inline __rtc_device PackedBSDF(const packedBSDF::NVisii  &nvisii)
      { type = TYPE_NVisii; data.nvisii = nvisii; }
      inline __rtc_device PackedBSDF(const packedBSDF::Glass  &glass)
      { type = TYPE_Glass; data.glass = glass; }
      inline __rtc_device PackedBSDF(const packedBSDF::Lambertian  &lambertian)
      { type = TYPE_Lambertian; data.lambertian = lambertian; }
      
      inline __rtc_device
      EvalRes eval(render::DG dg, vec3f w_i, bool dbg=false) const;

      inline __rtc_device
      float pdf(render::DG dg, vec3f w_i, bool dbg=false) const;
      
      inline __rtc_device
      void scatter(ScatterResult &scatter,
                   const render::DG &dg,
                   Random &random,
                   bool dbg=false) const;
      
      inline __rtc_device
      float getOpacity(bool isShadowRay,
                       bool isInMedium,
                       vec3f rayDir,
                       vec3f Ng,
                       bool dbg=false) const;
#endif
    };

#if RTC_DEVICE_CODE
    inline __rtc_device
    EvalRes PackedBSDF::eval(render::DG dg, vec3f w_i, bool dbg) const
    {
      if (type == TYPE_Phase)
        return data.phase.eval(dg,w_i,dbg);
      if (type == TYPE_NVisii)
        return data.nvisii.eval(dg,w_i,dbg);
      if (type == TYPE_Glass)
        return data.glass.eval(dg,w_i,dbg);
      if (type == TYPE_Lambertian)
        return data.lambertian.eval(dg,w_i,dbg);
      return EvalRes();
    }
    
    inline __rtc_device
    float PackedBSDF::pdf(render::DG dg, vec3f w_i, bool dbg) const
    {
      if (type == TYPE_NVisii)
        return data.nvisii.pdf(dg,w_i,dbg);
      if (type == TYPE_Glass)
        return data.glass.pdf(dg,w_i,dbg);
      if (type == TYPE_Lambertian)
        return data.lambertian.pdf(dg,w_i,dbg);
      if (type == TYPE_Phase)
        return data.phase.pdf(dg,w_i,dbg);
      return 0.f;
    }
    
    inline __rtc_device
    float PackedBSDF::getOpacity(bool isShadowRay,
                                 bool isInMedium,
                                 vec3f rayDir,
                                 vec3f Ng,
                                 bool dbg) const
    {
      if (type == TYPE_Glass)
        return data.glass.getOpacity(isShadowRay,isInMedium,rayDir,Ng,dbg);
      if (type == TYPE_NVisii)
        return data.nvisii.getOpacity(isShadowRay,isInMedium,rayDir,Ng,dbg);
      return 1.f;
    }

    inline __rtc_device
    void PackedBSDF::scatter(ScatterResult &scatter,
                             const render::DG &dg,
                             Random &random,
                             bool dbg) const
    {
      if (dbg) printf(" => scatter ...\n");
      scatter.pdf = 0.f;
      if (type == TYPE_Phase)
        return data.phase.scatter(scatter,dg,random,dbg);
      if (type == TYPE_NVisii)
        return data.nvisii.scatter(scatter,dg,random,dbg);
      if (type == TYPE_Glass)
        return data.glass.scatter(scatter,dg,random,dbg);
      if (type == TYPE_Lambertian)
        return data.lambertian.scatter(scatter,dg,random,dbg);
    }
#endif
  }
  using render::PackedBSDF;
}
