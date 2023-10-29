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

      /*! perform transfer function lookup for given scalar value, and
        return color/opacity-times-density value for that
        scalar. Mind: 'w' component of returend value is a *density*,
        not a alpha-opacity! */
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
    void buildParams(std::vector<OWLVarDecl> &params, size_t offset);
    void setParams(OWLLaunchParams lp);
    
    OWLBuffer           valuesBuffer = 0;
    range1f             domain;
    std::vector<vec4f>  values;
    float               baseDensity;
    DevGroup     *const devGroup;
  };






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
    int idx0_hi = clamp(int(f_hi)+1,0,numValues-2);
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
