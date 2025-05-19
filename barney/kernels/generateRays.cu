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
#include "barney/common/barney-common.h"
#include "barney/DeviceGroup.h"
#include "barney/render/Ray.h"
#include "barney/render/RayQueue.h"
#include "barney/Camera.h"
#include "barney/render/Renderer.h"
#include "barney/fb/FrameBuffer.h"
#include "rtcore/ComputeInterface.h"

namespace BARNEY_NS {
  namespace render {
    /*! generates a new wave-front of rays, to be written to
      'rayQueue[]', at (atomically incrementable) positoin
      *d_count. This kernel operates on *tiles* (not complete frames);
      the list of tiles to generate rays for is passed in 'tileDescs';
      there will be one cuda block per tile */
    struct GenerateRays {
#if RTC_DEVICE_CODE
      __rtc_device void run(const rtc::ComputeInterface &rtcore);
#endif
      
      Camera::DD camera;
      Renderer::DD renderer;
      /*! a unique random number seed value for pixel
        and lens jitter; probably just accumID */
      int accumID;
      /*! full frame buffer size, to check if a given
        tile's pixel ID is still valid */
      vec2i fbSize;
      /*! pointer to a device-side int that tracks the
        next write position in the 'write' ray
        queue; can be atomically incremented on the
        device */
      int *d_count;
      /*! pointer to device-side ray queue to write
        newly generated raysinto */
      SingleQueue rayQueue;
      /*! tile descriptors for the tiles that the
        frame buffer owns on this device; rays
        should only get generated for these tiles */
      TileDesc *tileDescs;
      bool enablePerRayDebug;
    };

#if RTC_DEVICE_CODE
    __rtc_device
    void GenerateRays::run(const rtc::ComputeInterface &rt)
    {
      // ------------------------------------------------------------------
      int tileID   = rt.getBlockIdx().x;
      int lPixelID = rt.getThreadIdx().x;

      vec2i tileOffset = tileDescs[tileID].lower;
      int ix = (lPixelID % tileSize) + tileOffset.x;
      int iy = (lPixelID / tileSize) + tileOffset.y;

      Ray ray;
      PathState state;
      state.misWeight = 0.f;
      state.pixelID = tileID * (tileSize*tileSize) + rt.getThreadIdx().x;
      Random rand(unsigned(ix+fbSize.x*accumID),
                  unsigned(iy+fbSize.y*accumID));
#if NEW_RNG
      ray.rngSeed.value = hash(ix,iy,accumID);
#else
      ray.rngSeed.seed(ix+fbSize.x*accumID,iy+fbSize.y*accumID);
#endif

      ray.org  = camera.lens_00;

      float pixel_u = ((accumID == 0) ? .5f : rand());
      float pixel_v = ((accumID == 0) ? .5f : rand());
      float image_u = ((ix+pixel_u)/float(fbSize.x));
      float image_v = ((iy+pixel_v)/float(fbSize.y));
      float aspect = fbSize.x / float(fbSize.y);
      vec3f ray_dir
        = camera.dir_00
        + (1.f*aspect*(image_u - .5f)) * camera.dir_du
        + (1.f*(image_v - .5f)) * camera.dir_dv;
      
      if (camera.apertureRadius > 0.f) {
        vec3f lens_du = normalize(camera.dir_du);
        vec3f lens_dv = normalize(camera.dir_dv);
        vec3f lensNormal  = cross(lens_du,lens_dv);

        vec3f D = normalize(ray_dir);
        vec3f pointOnImagePlane
          = D * (camera.focusDistance / fabsf(dot(D,lensNormal)));
        float lu, lv;
        if (accumID == 0) {
          lu = lv = 0.f;
        } else {
          while (true) {
            lu = 2.f*rand()-1.f;
            lv = 2.f*rand()-1.f;
            float f = lu*lu+lv*lv;
            if (f > 1.f) continue;
            break;
          }
        }
        vec3f lensOffset
          = (camera.apertureRadius * lu) * lens_du
          + (camera.apertureRadius * lv) * lens_dv;
        ray.org += lensOffset;
        ray_dir = normalize(pointOnImagePlane - lensOffset);
      } else {
        ray_dir = normalize(ray_dir);
      }
      ray.dir = ray_dir;
      
      bool crossHair_x = (ix == fbSize.x/2);
      bool crossHair_y = (iy == fbSize.y/2);

#ifdef NDEBUG
      ray.dbg         = 0;
#else
      ray.dbg         = enablePerRayDebug && (crossHair_x && crossHair_y);
#endif
      ray.clearHit();
      ray.isShadowRay = false;
      ray.isInMedium  = false;
      ray.tMax        = 1e30f;
      state.numDiffuseBounces = 0;
      if (0 && ray.dbg)
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
      
      vec4f bgColor
        = (renderer.bgColor.w >= 0.f)
        ? renderer.bgColor
        : ((1.0f - t)*vec4f(0.9f, 0.9f, 0.9f,1.f)
           + t *      vec4f(0.15f, 0.25f, .8f,1.f));
      if (renderer.bgTexture) {
        float bg_u = ((ix + pixel_u+.5f) / float(fbSize.x-1.f));
        float bg_v = ((iy + pixel_v+.5f) / float(fbSize.y-1.f));
        vec4f v = rtc::tex2D<vec4f>(renderer.bgTexture, bg_u, bg_v);//image_u + .5f / fbSize.x, image_v + .5f / fbSize.y);
        bgColor = v;
      }
      (vec4f&)ray.missColor = bgColor;
      if (0 && ray.dbg) printf("== spawn ray has bg tex %p bg color %f %f %f\n",
                               (void*)renderer.bgTexture,
                               bgColor.x,
                               bgColor.y,
                               bgColor.z);
      state.throughput = vec3f(1.f);
      int pos = rt.atomicAdd(d_count,1);
      
      rayQueue.rays[pos] = ray;
      rayQueue.states[pos] = state;
      rayQueue.hitIDs[pos] = {INFINITY,-1,-1,-1};
    }
#endif
  }  
  
  void Context::generateRays(Camera *camera,
                             Renderer *renderer,
                             FrameBuffer *fb)
  {
    auto getPerRayDebug = [&]()
    {
      const char *fromEnv = getenv("BARNEY_DBG_RENDER");
      return fromEnv && std::stoi(fromEnv);
    };
    static bool enablePerRayDebug = getPerRayDebug();
    
    assert(fb);
    int accumID=fb->accumID;
    // ------------------------------------------------------------------
    // launch all GPUs to do their stuff
    // ------------------------------------------------------------------
    Camera::DD cameraDD = camera->getDD();
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      TiledFB *devFB = fb->getFor(device);
      device->rayQueue->resetWriteQueue();
      render::GenerateRays args = {
        /* variable args */
        cameraDD,
        renderer->getDD(device),
        accumID,
        fb->numPixels,
        device->rayQueue->_d_nextWritePos,
        device->rayQueue->receiveAndShadeWriteQueue,
        devFB->tileDescs,
        enablePerRayDebug,
      };
      if (FromEnv::get()->logQueues) {
        std::stringstream ss;
        ss  << "#bn: ## ray queue op GENERATE " << device->rayQueue->receiveAndShadeWriteQueue.rays
            << " + " << device->rayQueue->receiveAndShadeWriteQueue.states
            << std::endl;
        std::cout << ss.str();
      }
      
      device->generateRays->launch(devFB->numActiveTiles,
                                   pixelsPerTile,
                                   &args);
    }
    // ------------------------------------------------------------------
    // wait for all GPUs' completion
    // ------------------------------------------------------------------
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      device->rtc->sync();
      device->rayQueue->swapAfterGeneration();
      device->rayQueue->numActive = device->rayQueue->readNumActive();
    }
  }
  
  RTC_EXPORT_COMPUTE1D(generateRays,BARNEY_NS::render::GenerateRays);
}


