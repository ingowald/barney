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


  typedef enum {
    RENDER_MODE_UNDEFINED,
    RENDER_MODE_LOCAL,
    RENDER_MODE_AO,
    RENDER_MODE_PT
  } RenderMode;
  
  __global__
  void g_shadeRays_local(AccumTile *accumTiles,
                      int accumID,
                      Ray *readQueue,
                      int numRays,
                      Ray *writeQueue,
                      int *d_nextWritePos,
                      int generation)
  {
    if (generation != 0) return;
    
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= numRays) return;

    Ray ray = readQueue[tid];
    // if (ray.dbg) printf("SHADE\n");
    
    vec3f albedo = (vec3f)ray.hit.baseColor;
    vec3f fragment = 0.f;
    float z = INFINITY;
    if (!ray.hadHit) {
      fragment = (vec3f)ray.hit.baseColor;
    } else {
      z = ray.tMax;
      vec3f dir = ray.dir;
      vec3f Ng = ray.hit.N;
      const bool isVolumeHit = (Ng == vec3f(0.f));
      if (!isVolumeHit) Ng = normalize(Ng);
      float NdotD = dot(Ng,normalize(dir));
      if (NdotD > 0.f) Ng = - Ng;
      
      // let's do some ambient eyelight-style shading, anyway:
      float scale
        = isVolumeHit
        ? .5f
        : (.2f + .4f*fabsf(NdotD));
      fragment
        = albedo
        * scale
        * ray.throughput;
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





  __global__
  void g_shadeRays_ao(AccumTile *accumTiles,
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
    if (0 && ray.dbg) {
      printf("============================================ ray: hadHit %i, t %f P %f %f %f base %f %f %f N %f %f %f\n",
             ray.hadHit,
             ray.tMax,
             (float)ray.hit.P.x,
             (float)ray.hit.P.y,
             (float)ray.hit.P.z,
             (float)ray.hit.baseColor.x,
             (float)ray.hit.baseColor.y,
             (float)ray.hit.baseColor.z,
             (float)ray.hit.N.x,
             (float)ray.hit.N.y,
             (float)ray.hit.N.z);
        }
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
      const bool isVolumeHit = (Ng == vec3f(0.f));
      if (!isVolumeHit) Ng = normalize(Ng);
      float NdotD = dot(Ng,normalize(dir));
      if (NdotD > 0.f) Ng = - Ng;
      
      // let's do some ambient eyelight-style shading, anyway:
      
      const float eyeLightWeight
        = isVolumeHit
        ? .5f
        : (.2f + .4f*fabsf(NdotD));
      const float ao_ambient_component = .1f;

      const float scale = ao_ambient_component * eyeLightWeight;
      // scale *= 0.001f;
      vec3f tp = ray.throughput;
      fragment
        = albedo
        * scale
        * ray.throughput;
      // if (ray.dbg) {
      //   printf("gen %i fragment %f %f %f\n",generation,fragment.x,fragment.y,fragment.z);
      //   printf("gen %i Ng %f %f %f\n",generation,Ng.x,Ng.y,Ng.z);
      //   printf("gen %i albedo %f %f %f\n",generation,albedo.x,albedo.y,albedo.z);
      //   printf("gen %i tp %f %f %f\n",generation,tp.x,tp.y,tp.z);
      // }
      
      // and then add a single diffuse bounce (ae, ambient occlusion)
      LCG<4> &rng = (LCG<4> &)ray.rngSeed;
      if (ray.hadHit && generation == 0) {
        Ray bounce;
        bounce.org = ray.hit.P + 1e-5f*Ng;
        // if (ray.dbg)
          // printf("bounce org %f %f %f\n",
          //        bounce.org.x,
          //        bounce.org.y,
          //        bounce.org.z);
        bounce.dir = normalize(Ng + randomDirection(rng));
        bounce.tMax = INFINITY;
        bounce.dbg = ray.dbg;
        bounce.hadHit = false;
        bounce.pixelID = ray.pixelID;
        rng();
        bounce.rngSeed = ray.rngSeed;
        rng();
        bounce.throughput = 
          // .6f *
          .8f *
          ray.throughput * albedo;
        writeQueue[atomicAdd(d_nextWritePos,1)] = bounce;
      }
    }
    int tileID  = ray.pixelID / pixelsPerTile;
    int tileOfs = ray.pixelID % pixelsPerTile;
    
    float4 &valueToAccumInto
      = accumTiles[tileID].accum[tileOfs];
    float  &tile_z
      = accumTiles[tileID].depth[tileOfs];
    vec4f valueToAccum = make_float4(fragment.x,fragment.y,fragment.z,0.f);

    // if (ray.dbg)
    //   printf("gen %i accumulating %f %f %f %f\n",
    //          generation,
    //          valueToAccum.x,
    //          valueToAccum.y,
    //          valueToAccum.z,
    //          valueToAccum.w);
    
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



  __global__
  void g_shadeRays_pt(AccumTile *accumTiles,
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
    if (0 && ray.dbg) {
      printf("============================================ ray: hadHit %i, t %f P %f %f %f base %f %f %f N %f %f %f\n",
             ray.hadHit,
             ray.tMax,
             (float)ray.hit.P.x,
             (float)ray.hit.P.y,
             (float)ray.hit.P.z,
             (float)ray.hit.baseColor.x,
             (float)ray.hit.baseColor.y,
             (float)ray.hit.baseColor.z,
             (float)ray.hit.N.x,
             (float)ray.hit.N.y,
             (float)ray.hit.N.z);
        }
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
      const bool isVolumeHit = (Ng == vec3f(0.f));
      if (!isVolumeHit) Ng = normalize(Ng);
      float NdotD = dot(Ng,normalize(dir));
      if (NdotD > 0.f) Ng = - Ng;
      
      // let's do some ambient eyelight-style shading, anyway:
      float scale
        = isVolumeHit
        ? .5f
        : (.2f + .4f*fabsf(NdotD));
      scale *= .01f;
      fragment
        = albedo
        * scale
        * ray.throughput;

      // and then add a single diffuse bounce (ae, ambient occlusion)
      LCG<4> &rng = (LCG<4> &)ray.rngSeed;
      if (ray.hadHit && generation < 5) {

        float transmission = (float)ray.hit.transmission;
        const bool doTransmission
          =  (transmission > 1e-4f)
          && (rng() < transmission);

        
        
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
        vec3f throughput
          = 
          // .6f *
          .8f *
          ray.throughput * albedo;
        float rr_prob = reduce_max(throughput);
        if (rng() <= rr_prob) {
          throughput = throughput * (1.f/rr_prob);
          bounce.throughput = throughput;
          writeQueue[atomicAdd(d_nextWritePos,1)] = bounce;
        }
      }
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

    static RenderMode renderMode = RENDER_MODE_UNDEFINED;
    if (renderMode == RENDER_MODE_UNDEFINED) {
      const char *_fromEnv = getenv("BARNEY_RENDER_MODE");
      if (!_fromEnv)
        _fromEnv = "AO";
      const std::string mode = _fromEnv;
      if (mode == "AO" || mode == "ao")
        renderMode = RENDER_MODE_AO;
      else if (mode == "PT" || mode == "pt")
        renderMode = RENDER_MODE_PT;
      else if (mode == "local")
        renderMode = RENDER_MODE_LOCAL;
      else
        throw std::runtime_error("unknown barney render mode '"+mode+"'");
    }
    
    if (nb) {
      switch(renderMode) {
      case RENDER_MODE_LOCAL:
        g_shadeRays_local<<<nb,bs,0,device->launchStream>>>
          (fb->accumTiles,fb->owner->accumID,
           rays.readQueue,numRays,rays.writeQueue,rays.d_nextWritePos,generation);
        break;
      case RENDER_MODE_AO:
        g_shadeRays_ao<<<nb,bs,0,device->launchStream>>>
          (fb->accumTiles,fb->owner->accumID,
           rays.readQueue,numRays,rays.writeQueue,rays.d_nextWritePos,generation);
        break;
      case RENDER_MODE_PT:
        g_shadeRays_pt<<<nb,bs,0,device->launchStream>>>
          (fb->accumTiles,fb->owner->accumID,
           rays.readQueue,numRays,rays.writeQueue,rays.d_nextWritePos,generation);
        break;
      }
    }
  }
  
}
