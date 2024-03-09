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

// some functions taken from OSPRay, under this lincense:
// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
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

#define pi (float(M_PI))
#define one_over_pi (float(1.f/M_PI))
#define two_pi (float(2.f*M_PI))

namespace barney {
  namespace render {

    inline __both__ float lerp(float factor, float a, float b) { return (1.f-factor)*a+factor*b; }
    inline __both__ vec3f lerp(vec3f factor, vec3f a, vec3f b) { return (1.f-factor)*a+factor*b; }

    inline __both__ float rcp(float f) { return 1.f/f; }
    inline __both__ float rcpf(float f) { return 1.f/f; }
    inline __both__ float abs(float f) { return fabsf(f); }

    inline __both__ float floor(float f) { return floorf(f); }
    inline __both__ float clamp(float f) { return min(1.f,max(0.f,f)); }
    inline __both__ float clamp(float f, float lo, float hi) { return min(hi,max(lo,f)); }
    inline __both__ float pow(float a, float b) { return powf(a,b); }
    inline __both__ float sqrt(float f) { return sqrtf(f); }
    inline __both__ float sqr(float f) { return f*f; }
    inline __both__ float cos2sin(const float f) { return sqrt(max(0.f, 1.f - sqr(f))); }
    inline __both__ float sin2cos(const float f) { return cos2sin(f); }

    
    inline __both__ float rcp_safe(float f) { return rcpf((fabsf(f) < 1e-8f) ? 1e-8f : f); }

    inline __both__ float interp1DLinear(float x, const float *f, int size)
    {
      float xc = clamp(x, 0.f, (float)(size-1));
      float s = xc - floor(xc);
      
      int x0 = min((int)xc, size-1);
      int x1 = min(x0+1,    size-1);
      
      return lerp(s, f[x0], f[x1]);
    }

    
    inline __device__ float interp3DLinear(vec3f p,
                                           float *f,
                                           vec3i size)
    { 
      float xc = clamp(p.x, 0.f, (float)(size.x-1));
      float yc = clamp(p.y, 0.f, (float)(size.y-1));
      float zc = clamp(p.z, 0.f, (float)(size.z-1));

      float sx = xc - floor(xc);
      float sy = yc - floor(yc);
      float sz = zc - floor(zc);

      int x0 = min((int)xc, size.x-1);
      int x1 = min(x0+1,    size.x-1);

      int y0 = min((int)yc, size.y-1);
      int y1 = min(y0+1,    size.y-1);

      int z0 = min((int)zc, size.z-1);
      int z1 = min(z0+1,    size.z-1);

      int ny = size.x;
      int nz = size.x * size.y;

      float f00 = lerp(sx, f[x0+y0*ny+z0*nz], f[x1+y0*ny+z0*nz]);
      float f01 = lerp(sx, f[x0+y1*ny+z0*nz], f[x1+y1*ny+z0*nz]);

      float f10 = lerp(sx, f[x0+y0*ny+z1*nz], f[x1+y0*ny+z1*nz]);
      float f11 = lerp(sx, f[x0+y1*ny+z1*nz], f[x1+y1*ny+z1*nz]);

      float f0 = lerp(sy, f00, f01);
      float f1 = lerp(sy, f10, f11);

      return lerp(sz, f0, f1);
    }


    
  }
}
