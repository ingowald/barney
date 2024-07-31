// ======================================================================== //
// Copyright 2022++ Ingo Wald                                               //
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

#pragma once

#include "owl/common/math/vec.h"
#include "owl/common/math/AffineSpace.h"

// #define DDA_FAST 1

namespace barney {

  namespace dda {
    using namespace owl::common;

#if DDA_FAST
    inline __device__ int get(vec3i v, int dim)
    {
      return dim == 0 ? v.x : (dim == 1 ? v.y : v.z);
    }
    inline __device__ float get(vec3f v, int dim)
    {
      return dim == 0 ? v.x : (dim == 1 ? v.y : v.z);
    }

    inline __device__ void set(vec3f &vec, int dim, float value)
    {
      vec.x = (dim == 0) ? value : vec.x;
      vec.y = (dim == 1) ? value : vec.y;
      vec.z = (dim == 2) ? value : vec.z;
    }
  
    inline __device__ void set(vec3i &vec, int dim, int value)
    {
      vec.x = (dim == 0) ? value : vec.x;
      vec.y = (dim == 1) ? value : vec.y;
      vec.z = (dim == 2) ? value : vec.z;
    }

    inline __device__ int smallestDim(vec3f v)
    {
      return v.x <= min(v.y,v.z) ? 0 : (v.y <= min(v.x,v.z) ? 1 : 2);
    }
#else
    inline __device__ int   get(vec3i v, int dim) { return v[dim]; }
    inline __device__ float get(vec3f v, int dim) { return v[dim]; }

    inline __device__ void set(vec3f &vec, int dim, float value) { vec[dim] = value; }
    inline __device__ void set(vec3i &vec, int dim, int   value) { vec[dim] = value; }

    inline __device__ int smallestDim(vec3f v)
    {
      return arg_min(v);
    }
#endif

    inline __device__ vec3f floor(vec3f v)
    {
      return { floorf(v.x),floorf(v.y),floorf(v.z) };
    }
  
    template<typename Lambda>
    inline __device__ void dda3(vec3f org,
                                vec3f dir,
                                float tMax,
                                vec3ui gridSize,
                                const Lambda &lambda,
                                bool dbg)
    {    
      const box3f bounds = { vec3f(0.f), vec3f(gridSize) };
      const vec3f floor_org = floor(org);
      const vec3f floor_org_plus_one = floor_org + vec3f(1.f);
      const vec3f rcp_dir     = rcp(dir);
      const vec3f abs_rcp_dir = abs(rcp(dir));
      const vec3f f_size = vec3f(gridSize);
    
      vec3f t_lo = (vec3f(0.f) - org) * rcp(dir);
      vec3f t_hi = (f_size     - org) * rcp(dir);
      vec3f t_nr = min(t_lo,t_hi);
      vec3f t_fr = max(t_lo,t_hi);
      if (dir.x == 0.f) {
        if (org.x < 0.f || org.x > f_size.x)
          // ray passes by the volume ...
          return;
        t_nr.x = -CUDART_INF; t_nr.x = +CUDART_INF;
      }
      if (dir.y == 0.f) {
        if (org.y < 0.f || org.y > f_size.y)
          // ray passes by the volume ...
          return;
        t_nr.y = -CUDART_INF; t_nr.y = +CUDART_INF;
      }
      if (dir.z == 0.f) {
        if (org.z < 0.f || org.z > f_size.z)
          // ray passes by the volume ...
          return;
        t_nr.z = -CUDART_INF; t_nr.z = +CUDART_INF;
      }
    
      float ray_t0 = max(0.f,reduce_max(t_nr));
      float ray_t1 = min(tMax,reduce_min(t_fr));
      // if (dbg) printf("t range for volume %f %f\n",ray_t0,ray_t1);
      if (ray_t0 > ray_t1) return; // no overlap with volume

    
      // compute first cell that ray is in:
      vec3f org_in_volume = org + ray_t0 * dir;
      vec3f f_cell = max(vec3f(0.f),min(f_size-1.f,floor(org_in_volume)));
      vec3f f_cell_end = {
        dir.x > 0.f ? f_cell.x+1.f : f_cell.x,
        dir.y > 0.f ? f_cell.y+1.f : f_cell.y,
        dir.z > 0.f ? f_cell.z+1.f : f_cell.z,
      };
      // if (dbg)
      //   printf("f_cell_end %f %f %f\n",
      //          f_cell_end.x,
      //          f_cell_end.y,
      //          f_cell_end.z);
    
      vec3f t_step = abs(rcp_dir);
      // if (dbg)
      //   printf("t_step %f %f %f\n",
      //          t_step.x,
      //          t_step.y,
      //          t_step.z);
      vec3f t_next
        = {
        ((dir.x == 0.f)
         ? CUDART_INF
         : (abs(f_cell_end.x - org_in_volume.x) * t_step.x)),
        ((dir.y == 0.f)
         ? CUDART_INF
         : (abs(f_cell_end.y - org_in_volume.y) * t_step.y)),
        ((dir.z == 0.f)
         ? CUDART_INF
         : (abs(f_cell_end.z - org_in_volume.z) * t_step.z))
      };
      // if (dbg)
      //   printf("t_next %f %f %f\n",
      //          t_next.x,
      //          t_next.y,
      //          t_next.z);
      const vec3i stop
        = {
        dir.x > 0.f ? (int)gridSize.x : -1,
        dir.y > 0.f ? (int)gridSize.y : -1,
        dir.z > 0.f ? (int)gridSize.z : -1
      };
      // if (dbg)
      //   printf("stop %i %i %i\n",
      //          stop.x,
      //          stop.y,
      //          stop.z);
      const vec3i cell_delta
        = {
        (dir.x > 0.f ? +1 : -1),
        (dir.y > 0.f ? +1 : -1),
        (dir.z > 0.f ? +1 : -1)
      };
      // if (dbg)
      //   printf("cell_delta %i %i %i\n",
      //          cell_delta.x,
      //          cell_delta.y,
      //          cell_delta.z);
      vec3i cell = vec3i(f_cell);
      float next_cell_begin = 0.f;
      while (1) {
        float t_closest = reduce_min(t_next);
        const float cell_t0 = ray_t0+next_cell_begin;
        const float cell_t1 = ray_t0+min(t_closest,tMax);
        // if (dbg)
        //   printf("cell %i %i %i dists %f %f %f closest %f t %f %f\n",
        //          cell.x,cell.y,cell.z,
        //          t_next.x,t_next.y,t_next.z,
        //          t_closest,cell_t0,cell_t1);
        bool wantToGoOn = lambda(cell,cell_t0,cell_t1);
        if (!wantToGoOn)
          return;
        next_cell_begin = t_closest;
        if (t_next.x == t_closest) {
          t_next.x += t_step.x;
          cell.x += cell_delta.x;
          if (cell.x == stop.x) return;
        }
        if (t_next.y == t_closest) {
          t_next.y += t_step.y;
          cell.y += cell_delta.y;
          if (cell.y == stop.y) return;
        }
        if (t_next.z == t_closest) {
          t_next.z += t_step.z;
          cell.z += cell_delta.z;
          if (cell.z == stop.z) return;
        }
      }
    }
  }

}
