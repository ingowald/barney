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

#include "owl/common/math/vec.h"

using namespace owl::common;

inline int randomInt() { return random(); }
inline int randomInt(int maxValue) { return (randomInt() % (maxValue+1)); }
inline int randomInt(int min, int max) { return min + randomInt(max-min); }

inline float randomFloat() { return drand48(); }
inline vec3f random3f() { return vec3f(randomFloat(),randomFloat(),randomFloat()); }
inline vec2f random2f() { return vec2f(randomFloat(),randomFloat()); }
inline float random1f() { return randomFloat(); }

