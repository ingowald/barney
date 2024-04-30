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

#include "barney/render/PackedBSDF.h"
#include "barney/render/HitAttributes.h"
// #include "barney/render/device/GeometryAttributes.h"
#include "materials/AnariMatte.h"
#include "materials/AnariPBR.h"

namespace barney {
  namespace render {
      
    struct DeviceMaterial {
      typedef enum {
        INVALID=0,
        TYPE_AnariMatte,
        TYPE_AnariPBF
      } Type;
      
      inline __device__
      PackedBSDF createBSDF(const HitAttributes &hitData) const;

      inline __device__
      void setHit(Ray &ray,
                  const HitAttributes &hitData) const;
        
      Type type;
      union {
        AnariPBR::DD   anariPBR;
        AnariMatte::DD anariMatte;
      };
    };

    inline __device__
    PackedBSDF DeviceMaterial::createBSDF(const HitAttributes &hitData) const
    {
      if (type == TYPE_AnariMatte)
        return anariMatte.createBSDF(hitData);
      return packedBSDF::Invalid();
    }

    // template<typename InterpolateGeometryAttribute>
    // inline __device__
    // void Material::setHit(Ray &ray,
    //                       HitAttributes      &hitAttribs,
    //                       // const GeometryAttributes &geomAttribs,
    //                       const InterpolateGeometryAttribute &interpolate)
    // {
    //   for (int i=0;i<numAttributes;i++)
    //     hitAttribs[i] = geomAttribs[i].
    //       printf("todo\n");
    // }

    inline __device__
    void DeviceMaterial::setHit(Ray &ray,
                                const HitAttributes &hitData) const
    {
      ray.setHit(hitData.worldPosition,hitData.worldNormal,
                 hitData.t,createBSDF(hitData));
    }
      
  }
}
