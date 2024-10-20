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
      __shared__ int l_count;
      if (threadIdx.x == 0)
        l_count = 0;

      // ------------------------------------------------------------------
      __syncthreads();
    
      int tileID = blockIdx.x;
    
      vec2i tileOffset = tileDescs[tileID].lower;
      int ix = (threadIdx.x % tileSize) + tileOffset.x;
      int iy = (threadIdx.x / tileSize) + tileOffset.y;

      Ray ray;
      ray.misWeight = 0.f;
      ray.pixelID = tileID * (tileSize*tileSize) + threadIdx.x;
      Random rand(ix+fbSize.x*accumID,
                  iy+fbSize.y*accumID);
      // Random rand(rngSeed,ray.pixelID);
      // rand();
      // rand();
      // rand();

      ray.org  = camera.lens_00;
      ray.dir
        = camera.dir_00
        + ((ix+((accumID==0)?.5f:rand()))/float(fbSize.x))*camera.dir_du
        + ((iy+((accumID==0)?.5f:rand()))/float(fbSize.y))*camera.dir_dv;
      
      if (camera.lensRadius > 0.f) {
        vec3f lens_du = normalize(camera.dir_du);
        vec3f lens_dv = normalize(camera.dir_dv);
        vec3f lensNormal  = cross(lens_du,lens_dv);

        vec3f D = normalize(ray.dir);
        vec3f pointOnImagePlane = D * (camera.focalLength / fabsf(dot(D,lensNormal)));
        float lu, lv;
        while (true) {
          lu = 2.f*rand()-1.f;
          lv = 2.f*rand()-1.f;
          float f = lu*lu+lv*lv;
          if (f > 1.f) continue;
          // lu *= 1.f/sqrtf(f);
          // lv *= 1.f/sqrtf(f);
          break;
        }
        vec3f lensOffset
          = (camera.lensRadius * lu) * lens_du
          + (camera.lensRadius * lv) * lens_dv;
        ray.org += lensOffset;
        ray.dir = normalize(pointOnImagePlane - lensOffset);
      } else {
        ray.dir = normalize(ray.dir);
      }
      
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
      vec3f bgColor = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
      // bool crossHair = 1 && (crossHair_x || crossHair_y);
      ray.missColor = bgColor*bgColor;
      // ray.hit.baseColor = .5f*ray.hit.baseColor*ray.hit.baseColor;
      // if (crossHair && !ray.dbg)
      //   ray.hit.baseColor = vec3f(1,0,0);
    
      // ray.hit.N = vec3f(0.f);
      ray.throughput = vec3f(1.f);
    
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
    }
  }
  
  void DeviceContext::generateRays_launch(TiledFB *fb,
                                          const Camera::DD &camera,
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
    
    render::g_generateRays
      <<<fb->numActiveTiles,pixelsPerTile,0,device->launchStream>>>
      (camera,
       rngSeed,
       fb->owner->accumID,
       fb->numPixels,
       rays._d_nextWritePos,
       rays.receiveAndShadeWriteQueue,
       fb->tileDescs,
       enablePerRayDebug);
  }
}
