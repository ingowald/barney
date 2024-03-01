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
#include <cstdlib>
#include <random>

using namespace owl::common;


inline int randomInt() { 
    return rand();
}
inline int randomInt(int maxValue) { return (randomInt() % (maxValue+1)); }
inline int randomInt(int min, int max) { return min + randomInt(max-min); }

inline float randomFloat() { 
    static std::random_device rd;  // a seed source for the random number engine
    static std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    static std::uniform_real_distribution<float> distrib(0.f,1.f);
	return distrib(gen); 
}

inline vec3f random3f() { return vec3f(randomFloat(),randomFloat(),randomFloat()); }
inline vec2f random2f() { return vec2f(randomFloat(),randomFloat()); }
inline float random1f() { return randomFloat(); }

