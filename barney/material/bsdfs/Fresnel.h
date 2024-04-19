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

// some functions taken from OSPRay, under this lincense:
// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
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

#include "../math.h"
#include "optics.h"
#include "GGXDistribution.h"

namespace barney {
  namespace render {
    
    //! \brief Computes fresnel coefficient for dielectric medium
    /*! \detailed Computes fresnel coefficient for media interface with
     *  relative refraction index eta. Eta is the outside refraction index
     *  divided by the inside refraction index. Both cosines have to be
     *  positive. */
    inline __both__ float fresnelDielectric(float cosI, float cosT, float eta)
    {
      const float Rper = (eta*cosI -     cosT) * rcp(eta*cosI +     cosT);
      const float Rpar = (    cosI - eta*cosT) * rcp(    cosI + eta*cosT);
      return 0.5f*(sqr(Rpar) + sqr(Rper));
    }

    /*! Computes fresnel coefficient for media interface with relative
     *  refraction index eta. Eta is the outside refraction index
     *  divided by the inside refraction index. The cosine has to be
     *  positive. */
    inline __both__ float fresnelDielectric(float cosI, float eta)
    {
      const float sqrCosT = render::sqrCosT(cosI, eta);
      if (sqrCosT < 0.0f) return 1.0f;
      return fresnelDielectric(cosI, sqrt(sqrCosT), eta);
    }

    inline __both__ float fresnelDielectricEx(float cosI, float &cosT, float eta)
    {
      const float sqrCosT = render::sqrCosT(cosI, eta);
      if (sqrCosT < 0.0f)
        {
          cosT = 0.0f;
          return 1.0f;
        }
      cosT = sqrt(sqrCosT);
      return fresnelDielectric(cosI, cosT, eta);
    }



    struct FresnelSchlick1 {
      // inline __device__ FresnelSchlick1() {}
      inline __device__
      FresnelSchlick1(vec3f r, float g) : r(r), g(g) {}
      
      inline __device__ vec3f eval(float cosI, bool dbg = false) const
      {
        const float c = 1.f - cosI;
        // if (dbg)
        //   printf("c %f r %f %f %f g %f\n",
        //          c,float(r.x),float(r.y),float(r.z), float(g));
        return lerp(sqr(sqr(c))*c, r, vec3f(g));
      }

      vec3f r; // reflectivity at normal incidence (0 deg)
      float g;// reflectivity at grazing angle (90 deg)
    };


    inline __device__
    float fresnelConductor(float cosI, float eta, float k)
    {
      const float tmp = sqr(eta) + sqr(k);
      const float Rpar
        = (tmp * sqr(cosI) - eta*(2.0f*cosI) + 1.f)
        * rcp(tmp * sqr(cosI) + eta*(2.0f*cosI) + 1.f);
      const float Rper
        = (tmp - 2.0f*eta*cosI + sqr(cosI))
        * rcp(tmp + 2.0f*eta*cosI + sqr(cosI));
      return 0.5f * (Rpar + Rper);
    }
    
    struct FresnelConductorRGBUniform
    {
      inline __device__
      FresnelConductorRGBUniform(vec3f eta, vec3f k)
        : eta(eta), k(k)
      {}

      inline __device__ vec3f eval(float cosI, bool dbg = false) const
      {
        return vec3f(fresnelConductor(cosI, this->eta.x, this->k.x),
                     fresnelConductor(cosI, this->eta.y, this->k.y),
                     fresnelConductor(cosI, this->eta.z, this->k.z));
      }
      
      vec3f eta;
      vec3f k;
    };

    
  }
}
