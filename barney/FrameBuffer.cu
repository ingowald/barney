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

#include "barney/FrameBuffer.h"

namespace barney {

  FrameBuffer::FrameBuffer(Context *context, const bool isOwner)
    : context(context),
      isOwner(isOwner)
  {
    assert(context);
    perDev.resize(context->devices.size());
    for (int localID=0;localID<context->devices.size();localID++) {
      perDev[localID]
        = TiledFB::create(context->getDevice(localID),this);
    }
  }

  void FrameBuffer::resize(vec2i size,
                           uint32_t *hostFB,
                           float    *hostDepth)
  {
    for (auto &pd: perDev)
      pd->resize(size);
    
    numPixels = size;

    if (isOwner) {
      // allocate/resize a owner-only, device-side depth buffer that
      // we can write into in device kernels
      if (finalDepth) {
        BARNEY_CUDA_CALL(Free(finalDepth));
        BARNEY_CUDA_CALL(MallocManaged(&finalDepth,
                                       numPixels.x*numPixels.y * sizeof(float)));
      }
      // save the host depth buffer, which may be host-accesible only
      this->hostDepth = hostDepth;
      
      if (finalFB && finalFB != this->hostFB) {
        BARNEY_CUDA_CALL(Free(finalFB));
        finalFB = nullptr;
      }
      
      this->hostFB = hostFB;
      cudaPointerAttributes attr;
      BARNEY_CUDA_CALL(PointerGetAttributes(&attr,hostFB));
      if (attr.type == cudaMemoryTypeHost) {
        BARNEY_CUDA_CALL(MallocManaged(&finalFB, numPixels.x*numPixels.y * sizeof(uint32_t)));
      } else {
        finalFB = hostFB;
      }
    }
  }
  
}
