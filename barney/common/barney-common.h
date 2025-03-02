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

// automatically generated, in build dir
#include "barney/api/common.h"
#include <owl/common/math/box.h>
#include <owl/common/math/AffineSpace.h>
#include <owl/common/math/random.h>
#ifdef __CUDACC__
# define OWL_DISABLE_TBB
#endif
#include <owl/common/parallel/parallel_for.h>
#include "rtcore/Frontend.h"
#include "rtcore/ComputeInterface.h"

// #include "barney/barney.h"
// #if BARNEY_HAVE_CUDA
// // #include "barney/common/cuda-helper.h"
// #include <owl/owl.h>
// #include <cuda_runtime.h>
// #endif
// #ifdef __CUDACC__
// #include <cuda/std/limits>
// #endif
// #if BARNEY_HAVE_CUDA
// #include <cuda.h>
// #endif

#define __barney_align(a) OWL_ALIGN(a)



// #ifdef __VECTOR_TYPES_H__
// // cuda/vector_types.h will define these types
// #else
// struct float2 { float x,y; };
// struct float3 { float x,y,z; };
// struct float4 { float x,y,z,w; };
// struct int2 { int x,y; };
// struct int3 { int x,y,z; };
// struct int4 { int x,y,z,w; };
// #endif

#if BARNEY_RTC_OPTIX
#  define BARNEY_NS barney_optix
#endif



namespace BARNEY_NS {
  // using namespace barney::rtc;
  
  using namespace owl::common;
  typedef owl::common::interval<float> range1f;
  using Random = LCG<8>;

  //  using rtc::load;
  
  template<typename T>
  inline __both__
  void swap(T &a, T &b) { T c = a; a = b; b = c; }

  // inline __both__ vec4f make_vec4f(float4 v) { return vec4f(v.x,v.y,v.z,v.w); }
  
  /*! helper function to extrace 3f spatial component from 4f point-plus-scalar */
  inline __both__ vec3f getPos(vec4f v)
  {return vec3f{v.x,v.y,v.z}; }

  /*! helper function to extrace 3f spatial component from 4f point-plus-scalar */
  // inline __both__ vec3f getPos(float4 v)
  // {return vec3f{v.x,v.y,v.z}; }

  /*! helper function to extrace 3f spatial component from 4f point-plus-scalar */
  inline __both__ box3f getBox(box4f bb)
  { return box3f{getPos(bb.lower),getPos(bb.upper)}; }
  inline __both__ box3f getBox(box3f bb)
  { return bb; }

  /*! helper function to extract 1f scalar range from 4f point-plus-scalar */
  inline __both__ range1f getRange(box4f bb)
  { return range1f{bb.lower.w,bb.upper.w}; }

}

#define BARNEY_NYI() throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" not yet implemented")

#define BARNEY_INVALID_VALUE() throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" invalid or un-implemented switch value")

