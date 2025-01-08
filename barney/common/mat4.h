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

#include "barney-common.h"
// std
#include <ostream>

namespace barney {

  struct mat4f {
    float e[16];

    __both__
    static mat4f identity()
    {
      mat4f res;
      
      res.e[ 0] = 1.f;
      res.e[ 1] = 0.f;
      res.e[ 2] = 0.f;
      res.e[ 3] = 0.f;

      res.e[ 4] = 0.f;
      res.e[ 5] = 1.f;
      res.e[ 6] = 0.f;
      res.e[ 7] = 0.f;

      res.e[ 8] = 0.f;
      res.e[ 9] = 0.f;
      res.e[10] = 1.f;
      res.e[11] = 0.f;

      res.e[12] = 0.f;
      res.e[13] = 0.f;
      res.e[14] = 0.f;
      res.e[15] = 1.f;
      return res;
    }
  };

  __both__
  inline vec4f operator*(const mat4f &m, const vec4f &v)
  {
    auto dot = [](vec4f a, vec4f b) {
      return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
    };
    return {
      dot(vec4f(m.e[ 0],m.e[ 4],m.e[ 8],m.e[12]),v),
      dot(vec4f(m.e[ 1],m.e[ 5],m.e[ 9],m.e[13]),v),
      dot(vec4f(m.e[ 2],m.e[ 6],m.e[10],m.e[14]),v),
      dot(vec4f(m.e[ 3],m.e[ 7],m.e[11],m.e[15]),v)
    };
  }

  inline std::ostream &operator<<(std::ostream &out, const mat4f &m)
  {
    out << '(' << m.e[ 0] << ',' << m.e[ 1] << ',' << m.e[ 2] << ',' << m.e[ 3] << ')'
        << '(' << m.e[ 4] << ',' << m.e[ 5] << ',' << m.e[ 6] << ',' << m.e[ 7] << ')'
        << '(' << m.e[ 8] << ',' << m.e[ 9] << ',' << m.e[10] << ',' << m.e[11] << ')'
        << '(' << m.e[12] << ',' << m.e[13] << ',' << m.e[14] << ',' << m.e[15] << ')';
    return out;
  }
} // barney

