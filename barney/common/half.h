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

/*! \file barney/half.h - "extends" owl::common:: to the half_t and vec3h
    types etc that it should have had in the first place */

#include "barney/common/barney-common.h"
#include <cuda_fp16.h>

namespace owl {
  namespace common  {
    struct vec3h {
      inline __both__ operator vec3f () const;
      inline __both__ vec3h &operator=(vec3f v);
    
      half x, y, z;
    };

    inline __both__ float from_half(half h) { return (float)h; }

    inline __both__ vec3f from_half(vec3h v)
    {
      return { from_half(v.x),from_half(v.y),from_half(v.z) };
    }

    inline __both__ half to_half(float f)
    {
      half h = f;
      return h;
    }

    inline __both__ vec3h to_half(vec3f v)
    {
      return { to_half(v.x),to_half(v.y),to_half(v.z) };
    }
  
    inline __both__ vec3h::operator vec3f () const
    {
      return from_half(*this);
    }
  
    inline __both__ vec3h &vec3h::operator=(vec3f v)
    {
      *this = to_half(v);
      return *this;
    }

    inline __both__ vec3f operator*(float f, vec3h v)  { return f * (vec3f)v; }
    inline __both__ vec3f operator*(vec3f a, vec3h b)  { return a * (vec3f)b; }
    inline __both__ vec3f operator*(vec3h a, vec3f b)  { return (vec3f)a * b; }
    inline __both__ vec3h operator*(vec3h a, vec3h b)  { return vec3h{ (float)a.x*(float)b.x,(float)a.y*(float)b.y,(float)a.z*(float)b.z}; }

    inline __both__ vec3f operator+(float f, vec3h v)  { return f + (vec3f)v; }
    inline __both__ vec3f operator+(vec3f a, vec3h b)  { return a + (vec3f)b; }
    inline __both__ vec3f operator+(vec3h a, vec3f b)  { return (vec3f)a + b; }

    inline __both__ vec3f normalize(vec3h v)    { return normalize((vec3f)v); }

    inline __both__ float dot(vec3h a, vec3h b)    { return dot((vec3f)a,(vec3f)b); }
    inline __both__ float dot(vec3h a, vec3f b)    { return dot((vec3f)a,(vec3f)b); }
    inline __both__ float dot(vec3f a, vec3h b)    { return dot((vec3f)a,(vec3f)b); }
    
  }
}



