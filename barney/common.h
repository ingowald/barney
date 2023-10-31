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

#include "owl/common/math/box.h"
#include "owl/common/math/random.h"
#include "owl/common/parallel/parallel_for.h"
#include "owl/owl.h"
#include "barney.h"
#include "barney/cuda-helper.h"
#include <cuda_runtime.h>
#include <string.h>
#include <mutex>
#include <vector>
#include <map>
#include <memory>

namespace barney {
  using namespace owl;
  using namespace owl::common;

  using range1f = interval<float>;

  using Random = LCG<4>;


  template<typename T>
  inline __device__
  void swap(T &a, T &b) { T c = a; a = b; b = c; }

  inline __device__
  float safeDiv(float a, float b) { return (b==0.f)?0.f:(a/b); }
  
}


