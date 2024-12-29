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

#ifdef __CUDACC__
# define OWL_DISABLE_TBB
#endif
#include "barney/DeviceContext.h"
#include "barney/render/Ray.h"
#include "barney/render/Renderer.h"
#include "barney/fb/FrameBuffer.h"

namespace barney {
  namespace render {
    /*! generates a new wave-front of rays, to be written to
      'rayQueue[]', at (atomically incrementable) positoin
      *d_count. This kernel operates on *tiles* (not complete frames);
      the list of tiles to generate rays for is passed in 'tileDescs';
      there will be one cuda block per tile */
    __global__
    void g_generateRays(/*! the camera used for generating the rays */
                        Camera::DD camera,
                        Renderer::DD renderer,
                        /*! a unique random number seed value for pixel
                          and lens jitter; probably just accumID */
                        int rngSeed,
                        int accumID,
                        /*! full frame buffer size, to check if a given
                          tile's pixel ID is still valid */
                        vec2i fbSize,
                        /*! pointer to a device-side int that tracks the
                          next write position in the 'write' ray
                          queue; can be atomically incremented on the
                          device */
                        int *d_count,
                        /*! pointer to device-side ray queue to write
                          newly generated raysinto */
                        Ray *rayQueue,
                        /*! tile descriptors for the tiles that the
                          frame buffer owns on this device; rays
                          should only get generated for these tiles */
                        TileDesc *tileDescs,
                        bool enablePerRayDebug)
    {
      // #define MERGE_ATOMICS 1 

#if MERGE_ATOMICS
      __shared__ int l_count;
      if (threadIdx.x == 0)
        l_count = 0;
#endif
      
      // ------------------------------------------------------------------
#if MERGE_ATOMICS
      __syncthreads();
#endif    
      int tileID = blockIdx.x;
    
      vec2i tileOffset = tileDescs[tileID].lower;
      int ix = (threadIdx.x % tileSize) + tileOffset.x;
      int iy = (threadIdx.x / tileSize) + tileOffset.y;

      Ray ray;
      ray.misWeight = 0.f;
      ray.pixelID = tileID * (tileSize*tileSize) + threadIdx.x;
      Random rand(ix+fbSize.x*accumID+ray.pixelID,
                  iy+fbSize.y*accumID);

      ray.org  = camera.lens_00;
      float image_u = ((ix+((accumID==0)?.5f:rand()))/float(fbSize.x));
      float image_v = ((iy+((accumID==0)?.5f:rand()))/float(fbSize.y));
      float aspect = fbSize.x / float(fbSize.y);
      vec3f ray_dir
        = camera.dir_00
        + (1.f*aspect*(image_u - .5f)) * camera.dir_du
        + (1.f*(image_v - .5f)) * camera.dir_dv;
      
      if (camera.lensRadius > 0.f) {
        vec3f lens_du = normalize(camera.dir_du);
        vec3f lens_dv = normalize(camera.dir_dv);
        vec3f lensNormal  = cross(lens_du,lens_dv);

        vec3f D = normalize(ray_dir);
        vec3f pointOnImagePlane = D * (camera.focalLength / fabsf(dot(D,lensNormal)));
        float lu, lv;
        while (true) {
          lu = 2.f*rand()-1.f;
          lv = 2.f*rand()-1.f;
          float f = lu*lu+lv*lv;
          if (f > 1.f) continue;
          break;
        }
        vec3f lensOffset
          = (camera.lensRadius * lu) * lens_du
          + (camera.lensRadius * lv) * lens_dv;
        ray.org += lensOffset;
        ray_dir = normalize(pointOnImagePlane - lensOffset);
      } else {
        ray_dir = normalize(ray_dir);
      }
      ray.dir = ray_dir;
      
      bool crossHair_x = (ix == fbSize.x/2);
      bool crossHair_y = (iy == fbSize.y/2);
 
      ray.dbg         = enablePerRayDebug && (crossHair_x && crossHair_y);
      ray.clearHit();
      ray.isShadowRay = false;
      ray.isInMedium  = false;
      ray.rngSeed     = rand.state;
      ray.tMax        = 1e30f;
      ray.numDiffuseBounces = 0;
      if (1 && ray.dbg)
        printf("-------------------------------------------------------\n");
      // if (ray.dbg)
      //   printf("  # generating INTO %lx\n",rayQueue);
             
      if (1 && ray.dbg)
        printf("======================\nspawned %f %f %f dir %f %f %f\n",
               ray.org.x,
               ray.org.y,
               ray.org.z,
               (float)ray.dir.x,
               (float)ray.dir.y,
               (float)ray.dir.z);

      const float t = (iy+.5f)/float(fbSize.y);
      // for *primary* rays we pre-initialize basecolor to a background
      // color; this way the shaderays function doesn't have to reverse
      // engineer pixel pos etc
      
      vec3f bgColor
        = (renderer.bgColor.w >= 0.f)
        ? (const vec3f&)renderer.bgColor
        : ((1.0f - t)*vec3f(0.9f, 0.9f, 0.9f) + t * vec3f(0.15f, 0.25f, .8f));
      if (renderer.bgTexture) {
        float4 v = tex2D<float4>(renderer.bgTexture,image_u,image_v);
        bgColor = (vec3f&)v;
      }
      ray.missColor = bgColor;
      if (ray.dbg) printf("== spawn ray has bg color %f %f %f\n",
                          bgColor.x,
                          bgColor.y,
                          bgColor.z);
      ray.throughput = vec3f(1.f);
    
#if MERGE_ATOMICS
      int pos = -1;
      if (ix < fbSize.x && iy < fbSize.y) 
        pos = atomicAdd(&l_count,1);

      // ------------------------------------------------------------------
      __syncthreads();
      if (threadIdx.x == 0) 
        l_count = atomicAdd(d_count,l_count);
    
      // ------------------------------------------------------------------
      __syncthreads();
      if (pos >= 0) 
        rayQueue[l_count + pos] = ray;
#else
      int pos = atomicAdd(d_count,1);
      rayQueue[pos] = ray;
#endif
    }
  }
  
  void DeviceContext::generateRays_launch(TiledFB *fb,
                                          const Camera::DD &camera,
                                          const Renderer::DD &renderer,
                                          int rngSeed)
  {
    auto device = fb->device;
    SetActiveGPU forDuration(device);

    auto getPerRayDebug = [&]()
    {
      const char *fromEnv = getenv("BARNEY_DBG_RENDER");
      return fromEnv && std::stoi(fromEnv);
    };
    static bool enablePerRayDebug = getPerRayDebug();

#if 1
    CHECK_CUDA_LAUNCH
      (/* cuda kernel */
       render::g_generateRays,
       /* launch config */
       fb->numActiveTiles,pixelsPerTile,0,device->launchStream,
       /* variable args */
       camera,
       renderer,
       rngSeed,
       fb->owner->accumID,
       fb->numPixels,
       rays._d_nextWritePos,
       rays.receiveAndShadeWriteQueue,
       fb->tileDescs,
       enablePerRayDebug);
#else
    render::g_generateRays
      <<<fb->numActiveTiles,pixelsPerTile,0,device->launchStream>>>
      (camera,
       renderer,
       rngSeed,
       fb->owner->accumID,
       fb->numPixels,
       rays._d_nextWritePos,
       rays.receiveAndShadeWriteQueue,
       fb->tileDescs,
       enablePerRayDebug);
#endif
  }
}
