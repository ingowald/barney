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

#pragma once

#include "mori/DeviceGroup.h"

namespace mori {

  struct Ray {
    vec3f    origin;
    vec3f    direction;
    float    tMax;
    int      instID, geomID, primID;
    float    u,v;
    uint32_t seed;
    // Payload  pay;
  };

  struct RayQueue {
    void init(DeviceContext *device)
    { this->device = device; }
    
    Ray *inQueue  = nullptr;
    Ray *outQueue = nullptr;
    int  numIn    = 0;
    int  numOut   = 0;
    int  size     = 0;

    DeviceContext *device = 0;
    
    void swap()
    {
      std::swap(inQueue, outQueue);
      std::swap(numIn,numOut);
    }

    void resize(int newSize)
    {
      SetActiveGPU forDuration(device);
      
      if (inQueue) MORI_CUDA_CALL(Free(inQueue));
      if (outQueue) MORI_CUDA_CALL(Free(outQueue));

      MORI_CUDA_CALL(Malloc(&inQueue,newSize*sizeof(Ray)));
      MORI_CUDA_CALL(Malloc(&outQueue,newSize*sizeof(Ray)));

      numIn = numOut = 0;
      size = newSize;
    }
    
  };
}
