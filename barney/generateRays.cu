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

  /*! generates a new wave-front of rays, to be written to
      'rayQueue[]', at (atomically incrementable) positoin
      *d_count. This kernel operates on *tiles* (not complete frames);
      the list of tiles to generate rays for is passed in 'tileDescs';
      there will be one cuda block per tile */
  __global__
  void g_generateRays(/*! the camera used for generating the rays */
                      Camera camera,
                      /*! a unique random number seed value for pixel
                          and lens jitter; probably just accumID */
                      int rngSeed,
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
                      TileDesc *tileDescs)
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

    // bool dbg = ((ix == 0) || (ix == fbSize.x-1)) && ((iy==0) || (iy == fbSize.y-1));
    
    Ray ray;
    ray.org  = camera.lens_00;
    ray.dir
      = camera.dir_00
      + (ix+.5f)*camera.dir_du
      + (iy+.5f)*camera.dir_dv;
    ray.dir = normalize(ray.dir);

    ray.centerPixel = ((ix == fbSize.x/2) && (iy == fbSize.y/2));
    ray.dbg = ray.centerPixel;
    // if (dbg) {
    //   vec3f ctr = normalize(camera.dbg_vi - camera.dbg_vp);
    //   float angle = dot(ctr,ray.direction);
    //   printf("pixel (%4i %4i) org %.1f %.1f %.1f dir %6.3f %6.3f %6.3f ctr %6.3f %6.3f %6.3f angle %f\n",
    //          ix,iy,
    //          ray.origin.x,
    //          ray.origin.y,
    //          ray.origin.z,
    //          ray.direction.x,
    //          ray.direction.y,
    //          ray.direction.z,ctr.x,ctr.y,ctr.z,angle);
    // }
    
    ray.tMax = 1e30f;
    ray.instID  = -1;
    ray.geomID  = -1;
    ray.primID  = -1;
    ray.u       = 0.f;
    ray.v       = 0.f;
    ray.pixelID = tileID * (tileSize*tileSize) + threadIdx.x;
    Random rand(rngSeed,ray.pixelID);
    ray.rngSeed = rand.state;
    ray.hadHit = false;

    const float t = (iy+.5f)/float(fbSize.y);
    ray.color = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
    
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
