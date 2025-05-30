// ======================================================================== //
// Copyright 2022 Stefan Zellmann                                           //
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

// Code in this file is adapted from OpenVKL. Original license follows

// Copyright 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "owl/common/math/vec.h"
#include "owl/common/math/LinearSpace.h"

namespace BARNEY_NS {

  inline __rtc_device
  vec3f make_vec3f(rtc::vec4f v) { return vec3f(v.x,v.y,v.z); }
  
  inline __rtc_device
  owl::common::LinearSpace3f make_LinearSpace3f(const owl::common::vec3f x,
                                                const owl::common::vec3f y,
                                                const owl::common::vec3f z)
  {
    owl::common::LinearSpace3f l;
    l.vx = x;
    l.vy = y;
    l.vz = z;
    return l;
  }

  inline __rtc_device
  float det(const owl::common::LinearSpace3f l)
  {
    return dot(l.vx, cross(l.vy, l.vz));
  }

  inline __rtc_device
  void pyramidInterpolationFunctions(float *pcoords/*[3]*/, float *sf/*[5]*/)
  {
    float rm, sm, tm;

    rm = 1.f - pcoords[0];
    sm = 1.f - pcoords[1];
    tm = 1.f - pcoords[2];

    sf[0] = rm * sm * tm;
    sf[1] = pcoords[0] * sm * tm;
    sf[2] = pcoords[0] * pcoords[1] * tm;
    sf[3] = rm * pcoords[1] * tm;
    sf[4] = pcoords[2];
  }

  inline __rtc_device
  void pyramidInterpolationDerivs(float *pcoords/*[3]*/,
                                  float *derivs/*[15]*/)
  {
    // r-derivatives
    derivs[0] = -(pcoords[1] - 1.f) * (pcoords[2] - 1.f);
    derivs[1] = (pcoords[1] - 1.f) * (pcoords[2] - 1.f);
    derivs[2] = pcoords[1] - pcoords[1] * pcoords[2];
    derivs[3] = pcoords[1] * (pcoords[2] - 1.f);
    derivs[4] = 0.f;

    // s-derivatives
    derivs[5] = -(pcoords[0] - 1.f) * (pcoords[2] - 1.f);
    derivs[6] = pcoords[0] * (pcoords[2] - 1.f);
    derivs[7] = pcoords[0] - pcoords[0] * pcoords[2];
    derivs[8] = (pcoords[0] - 1.f) * (pcoords[2] - 1.f);
    derivs[9] = 0.f;

    // t-derivatives
    derivs[10] = -(pcoords[0] - 1.f) * (pcoords[1] - 1.f);
    derivs[11] = pcoords[0] * (pcoords[1] - 1.f);
    derivs[12] = -pcoords[0] * pcoords[1];
    derivs[13] = (pcoords[0] - 1.f) * pcoords[1];
    derivs[14] = 1.f;
  }

  inline  __rtc_device
  bool intersectPyrEXT(float &value,
                       const vec3f &P,
                       const vec4f _v0,
                       const vec4f _v1,
                       const vec4f _v2,
                       const vec4f _v3,
                       const vec4f _v4
                       )
  {

    #define PYRAMID_DIVERGED               1e6f
    #define PYRAMID_MAX_ITERATION          10
    #define PYRAMID_CONVERGED              1e-4f
    #define PYRAMID_OUTSIDE_CELL_TOLERANCE 1e-6f

    const bool assumeInside = false;
    const float determinantTolerance = 1e-6f;
    const vec4f V[5] = {_v0,_v1,_v2,_v3,_v4};

    float pcoords[3] = {.5f, .5f, .5f};
    float derivs[15];
    float weights[5];

    bool converged = false;
    for (int iteration = 0;
         !converged && (iteration < PYRAMID_MAX_ITERATION);
          iteration++) {
      // Calculate element interpolation functions and derivatives
      pyramidInterpolationFunctions(pcoords, weights);
      pyramidInterpolationDerivs(pcoords, derivs);

      // Calculate newton functions
      vec3f fcol = 0.f, rcol = 0.f, scol = 0.f, tcol = 0.f;
      for (int i=0; i<5; ++i) {
        const vec3f pt = make_vec3f(V[i]);
        fcol = fcol + pt * weights[i];
        rcol = rcol + pt * derivs[i];
        scol = scol + pt * derivs[i + 5];
        tcol = tcol + pt * derivs[i + 10];
      }

      fcol = fcol - P;

      const float d = det(make_LinearSpace3f(rcol, scol, tcol));

      if (fabsf(d) < determinantTolerance) {
        return false;
      }

      const float d0 = det(make_LinearSpace3f(fcol, scol, tcol)) / d;
      const float d1 = det(make_LinearSpace3f(rcol, fcol, tcol)) / d;
      const float d2 = det(make_LinearSpace3f(rcol, scol, fcol)) / d;

      pcoords[0] = pcoords[0] - d0;
      pcoords[1] = pcoords[1] - d1;
      pcoords[2] = pcoords[2] - d2;

      // Convergence/divergence test - if neither, repeat
      if ((fabsf(d0) < PYRAMID_CONVERGED) & (fabsf(d1) < PYRAMID_CONVERGED) &
          (fabsf(d2) < PYRAMID_CONVERGED)) {
        converged = true;
      } else if ((fabsf(pcoords[0]) > PYRAMID_DIVERGED) |
                 (fabsf(pcoords[1]) > PYRAMID_DIVERGED) |
                 (fabsf(pcoords[2]) > PYRAMID_DIVERGED)) {
        return false;
      }
    }

    if (!converged) {
      return false;
    }

    const float lowerlimit = 0.f - PYRAMID_OUTSIDE_CELL_TOLERANCE;
    const float upperlimit = 1.f + PYRAMID_OUTSIDE_CELL_TOLERANCE;
    if (assumeInside || (pcoords[0] >= lowerlimit && pcoords[0] <= upperlimit &&
                         pcoords[1] >= lowerlimit && pcoords[1] <= upperlimit &&
                         pcoords[2] >= lowerlimit && pcoords[2] <= upperlimit)) {
      // Evaluation
      float val = 0.f;
      for (int i = 0; i < 5; i++) {
        val += weights[i] * V[i].w;
      }
      value = val;

      return true;
    }

    return false;
  }


  inline __rtc_device
  void primsInterpolationFunctions(float *pcoords/*[3]*/, float *sf/*[6]*/)
  {
    sf[0] = (1.f - pcoords[0] - pcoords[1]) * (1.f - pcoords[2]);
    sf[1] = pcoords[0] * (1.f - pcoords[2]);
    sf[2] = pcoords[1] * (1.f - pcoords[2]);
    sf[3] = (1.f - pcoords[0] - pcoords[1]) * pcoords[2];
    sf[4] = pcoords[0] * pcoords[2];
    sf[5] = pcoords[1] * pcoords[2];
  }

  inline __rtc_device
  void primsInterpolationDerivs(float *pcoords/*[3]*/, float *derivs/*[18]*/)
  {
    // r-derivatives
    derivs[0] = -1.f + pcoords[2];
    derivs[1] = 1.f - pcoords[2];
    derivs[2] = 0.f;
    derivs[3] = -pcoords[2];
    derivs[4] = pcoords[2];
    derivs[5] = 0.f;
  
    // s-derivatives
    derivs[6]  = -1.f + pcoords[2];
    derivs[7]  = 0.f;
    derivs[8]  = 1.f - pcoords[2];
    derivs[9]  = -pcoords[2];
    derivs[10] = 0.f;
    derivs[11] = pcoords[2];
  
    // t-derivatives
    derivs[12] = -1.f + pcoords[0] + pcoords[1];
    derivs[13] = -pcoords[0];
    derivs[14] = -pcoords[1];
    derivs[15] = 1.f - pcoords[0] - pcoords[1];
    derivs[16] = pcoords[0];
    derivs[17] = pcoords[1];
  }

  inline  __rtc_device
  bool intersectPrismEXT(float &value,
                         const vec3f &P,
                         const vec4f _v0,
                         const vec4f _v1,
                         const vec4f _v2,
                         const vec4f _v3,
                         const vec4f _v4,
                         const vec4f _v5)
  {

    #define PRIMS_DIVERGED               1e6f
    #define PRIMS_MAX_ITERATION          10
    #define PRIMS_CONVERGED              1e-4f
    #define PRIMS_OUTSIDE_CELL_TOLERANCE 1e-6f

    const bool assumeInside = false;
    const float determinantTolerance = 1e-6f;
    const vec4f V[6] = {_v0,_v1,_v2,_v3,_v4,_v5};

    float pcoords[3] = {.5f, .5f, .5f};
    float derivs[18];
    float weights[6];

    bool converged = false;
    for (int iteration = 0;
         !converged && (iteration < PRIMS_MAX_ITERATION);
         iteration++) {
      // Calculate element interpolation functions and derivatives
      primsInterpolationFunctions(pcoords, weights);
      primsInterpolationDerivs(pcoords, derivs);

      // Calculate newton functions
      vec3f fcol = 0.f, rcol = 0.f, scol = 0.f, tcol = 0.f;
      for (int i=0; i<6; ++i) {
        const vec3f pt = make_vec3f(V[i]);
        fcol = fcol + pt * weights[i];
        rcol = rcol + pt * derivs[i];
        scol = scol + pt * derivs[i + 6];
        tcol = tcol + pt * derivs[i + 12];
      }

      fcol = fcol - P;

      // Compute determinants and generate improvements
      const float d = det(make_LinearSpace3f(rcol, scol, tcol));

      if (fabsf(d) < determinantTolerance) {
        return false;
      }

      const float d0 = det(make_LinearSpace3f(fcol, scol, tcol)) / d;
      const float d1 = det(make_LinearSpace3f(rcol, fcol, tcol)) / d;
      const float d2 = det(make_LinearSpace3f(rcol, scol, fcol)) / d;

      pcoords[0] = pcoords[0] - d0;
      pcoords[1] = pcoords[1] - d1;
      pcoords[2] = pcoords[2] - d2;

      // Convergence/divergence test - if neither, repeat
      if ((fabsf(d0) < PRIMS_CONVERGED) & (fabsf(d1) < PRIMS_CONVERGED) &
          (fabsf(d2) < PRIMS_CONVERGED)) {
        converged = true;
      } else if ((fabsf(pcoords[0]) > PRIMS_DIVERGED) |
                 (fabsf(pcoords[1]) > PRIMS_DIVERGED) |
                 (fabsf(pcoords[2]) > PRIMS_DIVERGED)) {
        return false;
      }
    }

    if (!converged) {
      return false;
    }

    const float lowerlimit = 0.f - PRIMS_OUTSIDE_CELL_TOLERANCE;
    const float upperlimit = 1.f + PRIMS_OUTSIDE_CELL_TOLERANCE;
    if (assumeInside || (pcoords[0] >= lowerlimit && pcoords[0] <= upperlimit &&
                         pcoords[1] >= lowerlimit && pcoords[1] <= upperlimit &&
                         pcoords[2] >= lowerlimit && pcoords[2] <= upperlimit &&
                         pcoords[0] + pcoords[1] <= upperlimit)) {
      // Evaluation
      float val = 0.f;
      for (int i=0; i<6; ++i) {
        val += weights[i] * V[i].w;
      }
      value = val;

      return true;
    }

    return false;
  }


  inline __rtc_device
  void hexInterpolationFunctions(float *pcoords/*[3]*/, float *sf/*[8]*/)
  {
    float rm, sm, tm;
  
    rm = 1.f - pcoords[0];
    sm = 1.f - pcoords[1];
    tm = 1.f - pcoords[2];
  
    sf[0] = rm * sm * tm;
    sf[1] = pcoords[0] * sm * tm;
    sf[2] = pcoords[0] * pcoords[1] * tm;
    sf[3] = rm * pcoords[1] * tm;
    sf[4] = rm * sm * pcoords[2];
    sf[5] = pcoords[0] * sm * pcoords[2];
    sf[6] = pcoords[0] * pcoords[1] * pcoords[2];
    sf[7] = rm * pcoords[1] * pcoords[2];
  }

  inline __rtc_device
  void hexInterpolationDerivs(float *pcoords/*[3]*/, float *derivs/*[24]*/)
  {
    float rm, sm, tm;
  
    rm = 1.f - pcoords[0];
    sm = 1.f - pcoords[1];
    tm = 1.f - pcoords[2];
  
    // r-derivatives
    derivs[0] = -sm * tm;
    derivs[1] = sm * tm;
    derivs[2] = pcoords[1] * tm;
    derivs[3] = -pcoords[1] * tm;
    derivs[4] = -sm * pcoords[2];
    derivs[5] = sm * pcoords[2];
    derivs[6] = pcoords[1] * pcoords[2];
    derivs[7] = -pcoords[1] * pcoords[2];
  
    // s-derivatives
    derivs[8]  = -rm * tm;
    derivs[9]  = -pcoords[0] * tm;
    derivs[10] = pcoords[0] * tm;
    derivs[11] = rm * tm;
    derivs[12] = -rm * pcoords[2];
    derivs[13] = -pcoords[0] * pcoords[2];
    derivs[14] = pcoords[0] * pcoords[2];
    derivs[15] = rm * pcoords[2];
  
    // t-derivatives
    derivs[16] = -rm * sm;
    derivs[17] = -pcoords[0] * sm;
    derivs[18] = -pcoords[0] * pcoords[1];
    derivs[19] = -rm * pcoords[1];
    derivs[20] = rm * sm;
    derivs[21] = pcoords[0] * sm;
    derivs[22] = pcoords[0] * pcoords[1];
    derivs[23] = rm * pcoords[1];
  }

  inline __rtc_device
  bool intersectHexEXT(float &value,
                       const vec3f &P,
                       const vec4f v0,
                       const vec4f v1,
                       const vec4f v2,
                       const vec4f v3,
                       const vec4f v4,
                       const vec4f v5,
                       const vec4f v6,
                       const vec4f v7,
                       bool dbg=false)
  {
    #define HEX_DIVERGED               1e6f
    #define HEX_MAX_ITERATION          10
    #define HEX_CONVERGED              1e-4f
    #define HEX_OUTSIDE_CELL_TOLERANCE 1e-6f

    const bool assumeInside = false;
#if 1
    box3f b
      = box3f()
      .including((const vec3f&)v0)
      .including((const vec3f&)v1)
      .including((const vec3f&)v2)
      .including((const vec3f&)v3)
      .including((const vec3f&)v4)
      .including((const vec3f&)v5)
      .including((const vec3f&)v6)
      .including((const vec3f&)v7);
    
    const float determinantTolerance = 1e-10f * length(b.size());
#else
    const float determinantTolerance = 1e-6f;
#endif
    const vec4f V[8] = {v0,v1,v2,v3,v4,v5,v6,v7};

    float pcoords[3] = {0.5, 0.5, 0.5};
    float derivs[24];
    float weights[8];

    // Enter iteration loop
    bool converged = false;
    for (int iteration = 0; iteration < HEX_MAX_ITERATION;
         iteration++) {

      // Calculate element interpolation functions and derivatives
      hexInterpolationFunctions(pcoords, weights);
      hexInterpolationDerivs(pcoords, derivs);

      // Calculate newton functions
      vec3f fcol = 0.f, rcol = 0.f, scol = 0.f, tcol = 0.f;
      for (int i=0; i<8; ++i) {
        const vec3f pt = make_vec3f(V[i]);
        fcol = fcol + pt * weights[i];
        rcol = rcol + pt * derivs[i];
        scol = scol + pt * derivs[i + 8];
        tcol = tcol + pt * derivs[i + 16];
      }

      fcol = fcol - P;
      // Compute determinants and generate improvements

      const float d = det(make_LinearSpace3f(rcol, scol, tcol));
      if (fabsf(d) < determinantTolerance) {
        return false;
      }

      const float d0 = det(make_LinearSpace3f(fcol, scol, tcol)) / d;
      const float d1 = det(make_LinearSpace3f(rcol, fcol, tcol)) / d;
      const float d2 = det(make_LinearSpace3f(rcol, scol, fcol)) / d;

      pcoords[0] = pcoords[0] - d0;
      pcoords[1] = pcoords[1] - d1;
      pcoords[2] = pcoords[2] - d2;

      // Convergence/divergence test - if neither, repeat
      if ((fabsf(d0) < HEX_CONVERGED) & (fabsf(d1) < HEX_CONVERGED) &
          (fabsf(d2) < HEX_CONVERGED)) {
        converged = true;
        break;
      } else if ((fabsf(pcoords[0]) > HEX_DIVERGED) |
                 (fabsf(pcoords[1]) > HEX_DIVERGED) |
                 (fabsf(pcoords[2]) > HEX_DIVERGED)) {
        return false;
      }
    }

    if (!converged) {
      return false;
    }

    if (dbg) printf("converged : %i\n",int(converged));
    const float lowerlimit = 0.f - HEX_OUTSIDE_CELL_TOLERANCE;
    const float upperlimit = 1.f + HEX_OUTSIDE_CELL_TOLERANCE;
    if (assumeInside || (pcoords[0] >= lowerlimit && pcoords[0] <= upperlimit &&
                         pcoords[1] >= lowerlimit && pcoords[1] <= upperlimit &&
                         pcoords[2] >= lowerlimit && pcoords[2] <= upperlimit)) {
      // Evaluation
      float val = 0.f;
      for (int i=0; i<8; ++i) {
        val += weights[i] * V[i].w;
      }
      value = val;
      if (dbg) printf("hex value %f\n",val);
      return true;
    }

    return false;
  }

} // ::exa


