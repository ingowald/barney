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

#include "barney/DeviceContext.h"
#include "barney/Ray.h"

namespace barney {
  
  __global__
  void g_shadeRays(AccumTile *accumTiles,
                   Ray *readQueue,
                   int numRays,
                   Ray *writeQueue,
                   int *d_nextWritePos)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= numRays) return;

    Ray ray = readQueue[tid];
    vec3f color = abs(normalize(ray.direction));
    if (ray.hadHit)
      color = ray.color;
    int tileID  = ray.pixelID / pixelsPerTile;
    int tileOfs = ray.pixelID % pixelsPerTile;
    accumTiles[tileID].accum[tileOfs] = vec4f(color,0.f);
  }
  
}
