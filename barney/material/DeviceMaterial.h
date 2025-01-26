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

#include "barney/render/Ray.h"
#include "barney/packedBSDF/PackedBSDF.h"
#include "barney/render/HitAttributes.h"
// #include "barney/render/device/GeometryAttributes.h"
#include "barney/material/AnariMatte.h"
#include "barney/material/AnariPBR.h"

namespace barney {
  namespace render {
      
    struct DeviceMaterial {
      typedef enum {
        INVALID=0,
        TYPE_AnariMatte,
        TYPE_AnariPBR
      } Type;

      inline __both__
      PackedBSDF createBSDF(const HitAttributes &hitData,
                            const Sampler::DD *samplers,
                            bool dbg=false) const;
      inline __both__
      float getOpacity(const HitAttributes &hitData,
                       const Sampler::DD *samplers,
                       bool dbg) const;

      inline __both__
      void setHit(Ray &ray,
                  const HitAttributes &hitData,
                  const Sampler::DD *samplers,
                  bool dbg=false) const;
      
      Type type;
      union {
        AnariPBR::DD   anariPBR;
        AnariMatte::DD anariMatte;
      };
    };

    inline __both__
    PackedBSDF DeviceMaterial::createBSDF(const HitAttributes &hitData,
                                          const Sampler::DD *samplers,
                                          bool dbg) const
    {
      if (0 && dbg) printf("devicematerial type %i\n",(int)type);
      if (type == TYPE_AnariMatte)
        return anariMatte.createBSDF(hitData,samplers,dbg);
      if (type == TYPE_AnariPBR)
        return anariPBR.createBSDF(hitData,samplers,dbg);
#ifndef NDEBUG
      printf("#bn: DeviceMaterial::createBSDF encountered an invalid "
             "device material type (%i); most likely this is the app"
             " not having properly committed its material\n",(int)type);
#endif
      return packedBSDF::Invalid();
    }

    // inline __both__
    // float DeviceMaterial::getOpacity(const HitAttributes &hitData,
    //                                  const Sampler::DD *samplers,
    //                                  bool dbg) const
    // {
    //   PackedBSDF bsdf = createBSDF(hitData,samplers,dbg);
    //   return bsdf.getOpacity(hitData);
    //   // if (type == TYPE_AnariPBR)
    //   //   return anariPBR.getOpacity(isShadowRay,
    //   //                              hitData,samplers,dbg);
    //   // return 1.f;
    // }
    
    inline __both__
    void DeviceMaterial::setHit(Ray &ray,
                                const HitAttributes &hitData,
                                const Sampler::DD *samplers,
                                bool dbg) const
    {
      ray.setHit(hitData.worldPosition,hitData.worldNormal,
                 hitData.t,createBSDF(hitData,samplers,dbg));
    }
    
  }
}
