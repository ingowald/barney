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

#include "barney/fb/FrameBuffer.h"

namespace barney {

  FrameBuffer::FrameBuffer(Context *context, const bool isOwner)
    : Object(context),
      isOwner(isOwner)
  {
    perDev.resize(context->devices.size());
    for (int localID=0;localID<context->devices.size();localID++) {
      perDev[localID]
        = TiledFB::create(context->getDevice(localID),this);
    }
  }

  FrameBuffer::~FrameBuffer()
  {
    freeResources();
  }

  void FrameBuffer::freeResources()
  {
    if (finalDepth) {
      BARNEY_CUDA_CALL(Free(finalDepth));
      finalDepth = 0;
    }
    if (finalFB) {
      BARNEY_CUDA_CALL(Free(finalFB));
      finalFB = 0;
    }
  }
  
  void FrameBuffer::resize(vec2i size,
                           uint32_t *hostFB,
                           float    *hostDepth)
  {
    for (auto &pd: perDev)
      pd->resize(size);

    freeResources();
    numPixels = size;

#if 1
    if (isOwner) {
      // save the host pointers, which may be host-accesible only
      this->hostDepth = hostDepth;
      this->hostFB = hostFB;

      // allocate/resize a owner-only, device-side depth buffer that
      // we can write into in device kernels
      if (hostDepth) 
        // host wants a depth buffer, so we need to allocate one on
        // the device side for staging
        BARNEY_CUDA_CALL(Malloc(&finalDepth,
                                numPixels.x*numPixels.y*sizeof(float)));
      
      BARNEY_CUDA_CALL(Malloc(&finalFB, numPixels.x*numPixels.y*sizeof(uint32_t)));
    }
#else
    // original version before explicit 'freeResrouces()':
    if (isOwner) {
      // save the host pointers, which may be host-accesible only
      this->hostDepth = hostDepth;
      this->hostFB = hostFB;

      // allocate/resize a owner-only, device-side depth buffer that
      // we can write into in device kernels
      if (hostDepth) {
        // host wants a depth buffer, so we need to allocate one on
        // the device side for staging
        if (finalDepth) {
          BARNEY_CUDA_CALL(Free(finalDepth));
          finalDepth = 0;
        }
        BARNEY_CUDA_CALL(Malloc(&finalDepth,
                                numPixels.x*numPixels.y*sizeof(float)));
      }
      
      if (finalFB) {
        BARNEY_CUDA_CALL(Free(finalFB));
        finalFB = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&finalFB, numPixels.x*numPixels.y*sizeof(uint32_t)));
    }
#endif
  }
  
}
