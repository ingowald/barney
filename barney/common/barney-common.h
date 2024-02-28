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

#include <owl/common/math/box.h>
#include <owl/common/math/random.h>
#include <owl/common/parallel/parallel_for.h>
#include <owl/owl.h>
// #include "barney.h"
#include "barney/common/cuda-helper.h"
#include <cuda_runtime.h>
#include <string.h>
#include <mutex>
#include <vector>
#include <map>
#include <memory>
#include <sstream>
#include "barney.h"

#define __barney_align(a) OWL_ALIGN(a)

namespace barney {
  using namespace owl;
  using namespace owl::common;

  using range1f = interval<float>;

  using Random = LCG<6>;


  template<typename T>
  inline __device__
  void swap(T &a, T &b) { T c = a; a = b; b = c; }

  inline __device__
  float safeDiv(float a, float b) { return (b==0.f)?0.f:(a/b); }
  
  inline __both__ vec4f make_vec4f(float4 v) { return vec4f(v.x,v.y,v.z,v.w); }
  
  /*! helper functoin to extrace 3f spatial component from 4f point-plus-scalar */
  inline __both__ vec3f getPos(vec4f v)
  {return vec3f{v.x,v.y,v.z}; }

  /*! helper functoin to extrace 3f spatial component from 4f point-plus-scalar */
  inline __both__ vec3f getPos(float4 v)
  {return vec3f{v.x,v.y,v.z}; }

  /*! helper functoin to extrace 3f spatial component from 4f point-plus-scalar */
  inline __both__ box3f getBox(box4f bb)
  { return box3f{getPos(bb.lower),getPos(bb.upper)}; }
  inline __both__ box3f getBox(box3f bb)
  { return bb; }

  /*! helper functoin to extract 1f scalar range from 4f point-plus-scalar */
  inline __both__ range1f getRange(box4f bb)
  { return range1f{bb.lower.w,bb.upper.w}; }

  inline __both__ float lerp(float v0, float v1, float f)
  { return (1.f-f)*v0 + f*v1; }

  inline __both__ vec3f lerp(vec3f f, vec3f v0, vec3f v1)
  { return (vec3f(1.f)-f)*v0 + f*v1; }

  inline __both__ vec3f lerp(box3f box, vec3f f)
  { return lerp(f,box.lower,box.upper); }
  
  inline __both__ vec3f lerp(vec3f f, box3f box)
  { return lerp(f,box.lower,box.upper); }
}

#define BARNEY_NYI() throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" not yet implemented")

