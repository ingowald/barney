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

    typedef uint32_t BSDFType;

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

    inline __device__
    float luminance(vec3f c)
    { return 0.212671f*c.x + 0.715160f*c.y + 0.072169f*c.z; }


    struct BSDF {
      inline __device__ float importance() const { return luminance(albedo); }
      inline __device__ void init(vec3f albedo// , float importance
                                  )
      {
        this->albedo = albedo; //this->importance = importance;
      }
      vec3h albedo;
      // half  importance;
    };
    struct Minneart : public BSDF {
      inline __device__ void init(vec3f R, float b) { BSDF::init(R); this->b = b; }
      enum { bsdfType = BSDF_DIFFUSE_REFLECTION };
      /*! The amount of backscattering. A value of 0 means lambertian
       *  diffuse, and inf means maximum backscattering. */
      half b;
    };
    struct Velvety : public BSDF {
      inline __device__ void init(vec3f R, float f) { BSDF::init(R); this->f = f; }
      enum { bsdfType = BSDF_DIFFUSE_REFLECTION };
      /*! The falloff of horizon scattering. 0 no falloff,
       *  and inf means maximum falloff. */
      half f;
    };
    struct Velvet {
      struct MultiBSDF {
        inline __device__ vec3f albedo() const { return (vec3f)minneart.albedo+(vec3f)velvety.albedo; }
        // inline __device__ vec3f albedo() const { return minneart.albedo + velvety.albedo; }
        // inline __device__ int type() const { return minneart.type() | velvety.type(); }
        Minneart minneart;
        Velvety velvety;

        enum { bsdfType = Minneart::bsdfType | Velvety::bsdfType };
      };
      struct DD {
        inline __device__
        void make(MultiBSDF &multi)
        {
          multi.minneart.init(reflectance,backScattering);
          multi.velvety.init(horizonScatteringColor,horizonScatteringFallOff);
        }
        vec3f reflectance;
        vec3f horizonScatteringColor;
        float horizonScatteringFallOff;
        float backScattering;
      };
    };
    
  }
}
