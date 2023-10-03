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
#include <vector>
#include <map>
#include <mutex>
#include "mori/cuda-helper.h"

namespace mori {
  using namespace owl;
  using namespace owl::common;
  
  template<typename Payload>
  struct Ray {
    vec3f    origin;
    vec3f    direction;
    float    tMax;
    int      instID, geomID, primID;
    float    u,v;
    uint32_t seed;
    Payload  pay;
  };
  
}
