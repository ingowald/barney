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

namespace barney {
  namespace render {

#define BSDF_SPECULAR_REFLECTION   (1<<0)  /*!< perfect specular light reflection   */
#define BSDF_GLOSSY_REFLECTION     (1<<1)  /*!< glossy light reflection             */
#define BSDF_DIFFUSE_REFLECTION    (1<<2)  /*!< diffuse light reflection            */
#define BSDF_SPECULAR_TRANSMISSION (1<<3)  /*!< perfect specular light transmission */
#define BSDF_GLOSSY_TRANSMISSION   (1<<4)  /*!< glossy light transmission           */
#define BSDF_DIFFUSE_TRANSMISSION  (1<<5)  /*!< diffuse light transmission          */

/*! diffuse reflections and transmissions */
#define BSDF_DIFFUSE      (BSDF_DIFFUSE_REFLECTION   | BSDF_DIFFUSE_TRANSMISSION)

/*! glossy reflections and transmissions */
#define BSDF_GLOSSY       (BSDF_GLOSSY_REFLECTION    | BSDF_GLOSSY_TRANSMISSION)

/*! perfect specular reflections and transmissions */
#define BSDF_SPECULAR     (BSDF_SPECULAR_REFLECTION  | BSDF_SPECULAR_TRANSMISSION)

/*! all possible reflections */
#define BSDF_REFLECTION   (BSDF_DIFFUSE_REFLECTION   | BSDF_GLOSSY_REFLECTION   | BSDF_SPECULAR_REFLECTION)

/*! all possible transmissions */
#define BSDF_TRANSMISSION (BSDF_DIFFUSE_TRANSMISSION | BSDF_GLOSSY_TRANSMISSION | BSDF_SPECULAR_TRANSMISSION)

/*! all non-dirac types */
#define BSDF_SMOOTH       (BSDF_DIFFUSE | BSDF_GLOSSY)

/*! no component set */
#define BSDF_NONE         0

/*! all components set */
#define BSDF_ALL          (BSDF_REFLECTION | BSDF_TRANSMISSION)

    struct BSDF {
      inline __device__ BSDF() = default;
      inline __device__ BSDF(const BSDF &) = default;
      inline __device__ float importance() const { return luminance(albedo); }
      inline __device__ vec3f getAlbedo(bool dbg =false) const { return albedo; }
      inline __device__ BSDF(vec3f albedo=vec3f(.5f), bool dbg=false)
        : albedo(albedo)
      {}
      vec3f albedo;
      // half  importance;
    };

  }
}
