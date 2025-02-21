// ======================================================================== //
// Copyright 2025++ Ingo Wald                                               //
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

#include "barney/barneyConfig.h"
#include <cstring>
#include <cassert>
#include <mutex>
#include <vector>
#include <map>
#include <memory>
#include <sstream>
#include "barney/barney.h"
// #include "barney/api/mat4.h"

#include "owl/common/math/AffineSpace.h"

namespace barney_api {
  using namespace owl::common;

  typedef owl::common::interval<float> range1f;
}
