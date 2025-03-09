// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

#include "barney/DeviceGroup.h"
#include "barney/render/Ray.h"

namespace BARNEY_NS {
  using render::Ray;
  
  struct RayQueue {
    RayQueue(Device *device);
    ~RayQueue();
    int readNumActive();
    /*! how many rays are active in the *READ* queue */
    int numActiveRays() const;
    
    int *h_numActive = 0;

    /*! the read queue, where local kernels operating on rays (trace
      and shade) can read rays from. this is actually a misnomer
      becasue both shade and trace can actually modify trays (and
      thus, strictly speaking, are 'writing' to those rays), but
      haven't yet found a better name */
    Ray *traceAndShadeReadQueue  = nullptr;

    /*! the queue where local kernels that write *new* rays
      (ie, ray gen and shading) will write their rays into */
    Ray *receiveAndShadeWriteQueue = nullptr;

    /*! current write position in the write queue (during shading and
      ray generation) */
    int *_d_nextWritePos  = 0;
    
    /*! how many rays are active in the *READ* queue */
    int  numActive = 0;
    int  size      = 0;

    Device *device = 0;

    void resetWriteQueue();
      
    void swap();

    void reserve(int requiredSize);
    void resize(int newSize);
    
  };

}
