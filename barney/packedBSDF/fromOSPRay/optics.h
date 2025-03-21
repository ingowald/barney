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

namespace BARNEY_NS {
  namespace render {
    // helper function which computes cosT^2 from cosI and eta
    inline __both__ float sqrCosT(const float cosI, const float eta)
    {
      return 1.0f - sqr(eta)*(1.0f - sqr(cosI));
    }

    /*! Reflects a viewing vector I at a normal N. Cosine between I
     *  and N is given as input. */
    inline __both__ vec3f reflect(const vec3f& I, const vec3f& N, float cosI)
    {
      return (2.0f*cosI) * N - I;
    }

#if 0
    /*! Reflects a viewing vector I at a normal N. */
    inline __both__  vec3f reflect(const vec3f& I, const vec3f& N)
    {
      return reflect(I, N, dot(I, N));
    }
#endif

    //! \brief Refracts a viewing vector I at a normal N
    /*! \detailed Refracts a viewing vector I at a normal N using the
     *  relative refraction index eta. Eta is refraction index of outside
     *  medium (where N points into) divided by refraction index of the
     *  inside medium. The vectors I and N have to point towards the same
     *  side of the surface. The cosine between I and N, and the cosine of -N and
     *  the refracted ray is given as input */
    inline  __both__ vec3f refract(const vec3f& I, const vec3f& N, float cosI, float cosT, float eta)
    {
      return eta*(cosI*N - I) - cosT*N;
    }

    inline  __both__ vec3f refract(const vec3f& I, const vec3f& N, float cosI, float eta)
    {
      const float sqrCosT = render::sqrCosT(cosI, eta);
      if (sqrCosT < 0.0f) return vec3f(0.f);
      return refract(I, N, cosI, sqrtf(sqrCosT), eta);
    }

    inline  __both__ float refract(float cosI, float eta)
    {
      const float sqrCosT = render::sqrCosT(cosI, eta);
      return sqrtf(max(sqrCosT, 0.0f));
    }

  }
}

