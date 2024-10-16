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

#include "barney/render/DG.h"
// #include "packedBSDFs/VisRTX.h"
#include "barney/packedBSDF/NVisii.h"
#include "barney/packedBSDF/fromOSPRay/Glass.h"
#include "barney/packedBSDF/Phase.h"

namespace barney {
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
        TYPE_NVisii
      } Type;
      struct Data {
        union {
          packedBSDF::Phase  phase;
          // packedBSDF::VisRTX visRTX;
          packedBSDF::Glass  glass;
          packedBSDF::NVisii nvisii;
        };
      } data;

      Type type;

#ifdef __CUDACC__
      inline __device__ PackedBSDF();
      inline __device__ PackedBSDF(Type type, Data data)
        : type(type), data(data) {}
      inline __device__ PackedBSDF(const packedBSDF::Invalid &invalid)
      { type = INVALID; }
      inline __device__ PackedBSDF(const packedBSDF::Phase  &phase)
      { type = TYPE_Phase; data.phase = phase; }
      inline __device__ PackedBSDF(const packedBSDF::NVisii  &nvisii)
      { type = TYPE_NVisii; data.nvisii = nvisii; }
      inline __device__ PackedBSDF(const packedBSDF::Glass  &glass)
      { type = TYPE_Glass; data.glass = glass; }
      // inline __device__ PackedBSDF(const packedBSDF::VisRTX &visRTX);
      
      inline __device__
      EvalRes eval(render::DG dg, vec3f w_i, bool dbg=false) const;

      inline __device__
      float pdf(render::DG dg, vec3f w_i, bool dbg=false) const;
      
      inline __device__
      void scatter(ScatterResult &scatter,
                   const render::DG &dg,
                   Random &random,
                   bool dbg=false) const;
      
      inline __device__
      float getOpacity(bool isShadowRay,
                       bool isInMedium,
                       vec3f rayDir,
                       vec3f Ng,
                       bool dbg=false) const;
#endif
    };


#ifdef __CUDACC__
    // inline __device__
    // PackedBSDF::PackedBSDF(const packedBSDF::VisRTX &visRTX)
    // { type = TYPE_VisRTX; data.visRTX = visRTX; }
    
    inline __device__
    EvalRes PackedBSDF::eval(render::DG dg, vec3f w_i, bool dbg) const
    {
      if (type == TYPE_Phase)
        return data.phase.eval(dg,w_i,dbg);
      // if (type == TYPE_VisRTX)
      //   return data.visRTX.eval(dg,w_i,dbg);
      if (type == TYPE_NVisii)
        return data.nvisii.eval(dg,w_i,dbg);
      if (type == TYPE_Glass)
        return data.glass.eval(dg,w_i,dbg);
      return EvalRes();
    }
    
    inline __device__
    float PackedBSDF::pdf(render::DG dg, vec3f w_i, bool dbg) const
    {
      if (type == TYPE_NVisii)
        return data.nvisii.pdf(dg,w_i,dbg);
      if (type == TYPE_Glass)
        return data.glass.pdf(dg,w_i,dbg);
      return 0.f;
    }
    
    inline __device__
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

    inline __device__
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
    }
#endif
    
  }
}
