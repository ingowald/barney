// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

#include "barney/DeviceGroup.h"

namespace barney {

  struct TransferFunction {
    struct DD {

      static void addVars(std::vector<OWLVarDecl> &vars, int base);
      
      /*! maps given scalar through this trnasfer function, and
        returns /opacity-times-density value for that scalar. scalars
        outside the specified domain get mapped to the boundary
        values; values inside the domain get linearly interpolated
        from the array of values.  Mind: 'w' component of returned
        value is a *density*, not a alpha-opacity! */
      inline __device__
      vec4f map(float s, bool dbg=false) const;

      /*! compute the majorant (max opacity times baseDensity) for
          given range of scalar values */
      inline __device__
      float majorant(range1f r, bool dbg = false) const;

      float4  *values;
      range1f  domain;
      float    baseDensity;
      int      numValues;
    };

    TransferFunction(DevGroup *devGroup);

    /*! get cuda-usable device-data for given device ID (relative to
        devices in the devgroup that this gris is in */
    DD getDD(int devID) const;
    
    void set(const range1f &domain,
             const std::vector<vec4f> &values,
             float baseDensity);
    std::vector<OWLVarDecl> getVarDecls(uint32_t myOffset);
    void setVariables(OWLGeom geom) const;
    
    OWLBuffer           valuesBuffer = 0;
    range1f             domain = { 0.f, 1.f };
    std::vector<vec4f>  values;
    float               baseDensity;
    DevGroup     *const devGroup;
  };



  /*! maps given scalar through this trnasfer function, and returns
    /opacity-times-density value for that scalar. scalars outside the
    specified domain get mapped to the boundary values; values inside
    the domain get linearly interpolated from the array of values.
    Mind: 'w' component of returned value is a *density*, not a
    alpha-opacity! */
  inline __device__
  vec4f TransferFunction::DD::map(float s, bool dbg) const
  {
    float f = (s-domain.lower)/domain.span();
    f = clamp(f,0.f,1.f);
    f *= (numValues-1);
    int idx = clamp(int(f),0,numValues-2);
    f -= idx;

    float4 v0 = values[idx];
    float4 v1 = values[idx+1];
    vec4f r = (1.f-f)*(const vec4f&)v0 + f*(const vec4f&)v1;
    r.w *= baseDensity;
    return r;
  }


  /*! compute the majorant (max opacity times baseDensity) for
    given range of scalar values */
  inline __device__
  float TransferFunction::DD::majorant(range1f r, bool dbg) const
  {
    float f_lo = (r.lower-domain.lower)/domain.span();
    float f_hi = (r.upper-domain.lower)/domain.span();
    f_lo = clamp(f_lo,0.f,1.f);
    f_hi = clamp(f_hi,0.f,1.f);
    f_lo *= (numValues-1);
    f_hi *= (numValues-1);
    int idx0_lo = clamp(int(f_lo),0,numValues-2);
    int idx0_hi = clamp(int(f_hi),0,numValues-2);
    f_lo -= idx0_lo;
    f_hi -= idx0_hi;

    float lerp_lo = (1.f-f_lo)*values[idx0_lo].w + f_lo*values[idx0_lo+1].w;
    float lerp_hi = (1.f-f_hi)*values[idx0_hi].w + f_hi*values[idx0_hi+1].w;
    
    float m = 0.f;
    m = max(m,lerp_lo);
    m = max(m,lerp_hi);
    for (int i=idx0_lo+1;i<=idx0_hi;i++)
      m = max(m,values[i].w);
    return m * baseDensity;
  }
  
}
