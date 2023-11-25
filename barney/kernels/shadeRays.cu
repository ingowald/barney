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
#include "barney/fb/FrameBuffer.h"
#include "barney/fb/TiledFB.h"

namespace barney {

  inline __device__
  vec3f randomDirection(Random &rng)
  {
    vec3f v;
    while (true) {
      v.x = 1.f-2.f*rng();
      v.y = 1.f-2.f*rng();
      v.z = 1.f-2.f*rng();
      if (dot(v,v) <= 1.f)
        return normalize(v);
    }
  }
  
  __global__
  void g_shadeRays(AccumTile *accumTiles,
                   int accumID,
                   Ray *readQueue,
                   int numRays,
                   Ray *writeQueue,
                   int *d_nextWritePos,
                   int generation)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= numRays) return;

    Ray ray = readQueue[tid];
    
    vec3f albedo = (vec3f)ray.hit.baseColor;
    vec3f fragment = 0.f;
    float z = INFINITY;
    if (!ray.hadHit) {
      if (generation == 0) {
        // for primary rays we have pre-initialized basecolor to a
        // background color in generateRays(); let's just use this, so
        // generaterays can pre--set whatever color it wasnts for
        // non-hitting rays
        fragment = (vec3f)ray.hit.baseColor;
      } else {
        vec3f ambientIllum = vec3f(1.f);
        fragment = ray.throughput * ambientIllum;
      }
    } else {
      z = ray.tMax;
      vec3f dir = ray.dir;
      vec3f Ng = ray.hit.N;
      float NdotD = dot(Ng,dir);
      if (NdotD > 0.f) Ng = - Ng;

      // let's do some ambient eyelight-style shading, anyway:
      float scale = .2f + .4f*fabsf(NdotD);
      scale *= .3f;
      fragment
        = albedo
        * scale
        * ray.throughput;

      // and then add a single diffuse bounce (ae, ambient occlusion)
#if 1
      LCG<4> &rng = (LCG<4> &)ray.rngSeed;
      if (ray.hadHit && generation == 0) {
        Ray bounce;
        bounce.org = ray.hit.P + 1e-3f*Ng;
        bounce.dir = normalize(Ng + randomDirection(rng));
        bounce.tMax = INFINITY;
        bounce.dbg = ray.dbg;
        bounce.hadHit = false;
        bounce.pixelID = ray.pixelID;
        rng();
        bounce.rngSeed = ray.rngSeed;
        rng();
        bounce.throughput = .6f * ray.throughput * albedo;

#if PRINT_BALLOT
        int numActive = __popc(__ballot(1));
        
        if (ray.dbg) printf("=================================\n**** NEW *SECONDARY* RAY, numActive = %i\n", numActive);
        bounce.numPrimsThisRay = 0;
        bounce.numIsecsThisRay = 0;
        bounce.numLeavesThisRay = 0;
#endif
        writeQueue[atomicAdd(d_nextWritePos,1)] = bounce;
      }
#endif
    }
    int tileID  = ray.pixelID / pixelsPerTile;
    int tileOfs = ray.pixelID % pixelsPerTile;
    
    float4 &valueToAccumInto
      = accumTiles[tileID].accum[tileOfs];
    float  &tile_z
      = accumTiles[tileID].depth[tileOfs];
    vec4f valueToAccum = make_float4(fragment.x,fragment.y,fragment.z,0.f);
    if (accumID > 0)
      valueToAccum = valueToAccum + (vec4f)valueToAccumInto;
    
    if (generation == 0) {
      if (accumID == 0)
        tile_z = z;
      else
        tile_z = min(tile_z,z);
    }
        
    valueToAccumInto = valueToAccum;
  }
  
  void DeviceContext::shadeRays_launch(TiledFB *fb,
                                       int generation)
  {
    SetActiveGPU forDuration(device);
    int numRays = rays.numActive;
    int bs = 1024;
    int nb = divRoundUp(numRays,bs);

    if (nb)
      g_shadeRays<<<nb,bs,0,device->launchStream>>>
        (fb->accumTiles,fb->owner->accumID,
         rays.readQueue,numRays,rays.writeQueue,rays.d_nextWritePos,generation);
  }

}
