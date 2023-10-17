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
    moris.resize(context->gpuIDs.size());
    for (int localID=0;localID<context->gpuIDs.size();localID++) {
      moris[localID]
        = mori::TiledFB::create(context->moris[localID]->device);
    }
  }

  void FrameBuffer::resize(vec2i size, uint32_t *hostFB)
  {
    for (auto &mori: moris)
      mori->resize(size);
    
    numPixels = size;
    // numTiles  = divRoundUp(size,vec2i(mori::tileSize));

    if (isOwner) {
      if (finalFB && finalFB != this->hostFB) {
        MORI_CUDA_CALL(Free(finalFB));
        finalFB = nullptr;
      }
      
      this->hostFB = hostFB;
      cudaPointerAttributes attr;
      MORI_CUDA_CALL(PointerGetAttributes(&attr,hostFB));
      if (attr.type == cudaMemoryTypeHost) {
        MORI_CUDA_CALL(MallocManaged(&finalFB, numPixels.x*numPixels.y * sizeof(uint32_t)));
        std::cout << "#### OWNER ALLOCED FINAL FB " << finalFB << " of size << " << numPixels << std::endl;
      } else {
        std::cout << "### simply using host-supplied frame buffer " << hostFB << std::endl;
        finalFB = hostFB;
      }
      fflush(0);
    }
  }
  
}
