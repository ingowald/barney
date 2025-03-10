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

#include "RayQueue.h"

namespace BARNEY_NS {

  RayQueue::RayQueue(Device *device)
    : device(device)
  {
    // BARNEY_CUDA_CALL(MallocHost((void **)&h_numActive,sizeof(int)));
    auto rtc = device->rtc;
    h_numActive = (int*)rtc->allocHost(sizeof(int));
  }
    
  RayQueue::~RayQueue()
  {
    // BARNEY_CUDA_CALL(FreeHost(h_numActive));
    auto rtc = device->rtc;
    rtc->freeHost(h_numActive);
  }

  int RayQueue::readNumActive()
  {
    auto rtc = device->rtc;
    rtc->copyAsync(h_numActive,_d_nextWritePos,sizeof(int));
    rtc->sync();
    // BARNEY_CUDA_CALL(MemcpyAsync(h_numActive,_d_nextWritePos,sizeof(int),
    //                              cudaMemcpyDeviceToHost,
    //                              device->launchStream));
    // BARNEY_CUDA_CALL(StreamSynchronize(device->launchStream));
    numActive = *h_numActive;
    return *h_numActive;
  }
    
  /*! how many rays are active in the *READ* queue */
  int RayQueue::numActiveRays() const
  {
    return numActive;
  }
    
  void RayQueue::resetWriteQueue()
  {
    auto rtc = device->rtc;
    rtc->memsetAsync(_d_nextWritePos,0,sizeof(int));
  }
    
  void RayQueue::swap()
  {
    std::swap(receiveAndShadeWriteQueue, traceAndShadeReadQueue);
  }

  void RayQueue::reserve(int requiredSize)
  {
    assert(this);
    if (size >= requiredSize) return;
    resize(requiredSize);
  }
    
  void RayQueue::resize(int newSize)
  {
    SetActiveGPU forDuration(device);
    auto rtc = device->rtc;
    if (traceAndShadeReadQueue)
      rtc->freeMem(traceAndShadeReadQueue);
    if (receiveAndShadeWriteQueue)
      rtc->freeMem(receiveAndShadeWriteQueue);

    if (!_d_nextWritePos) {
      _d_nextWritePos = (int*)rtc->allocMem(sizeof(int)); 
    }

    traceAndShadeReadQueue = (Ray*)rtc->allocMem(newSize*sizeof(Ray));
    receiveAndShadeWriteQueue = (Ray*)rtc->allocMem(newSize*sizeof(Ray));
        
    size = newSize;

    resetWriteQueue();
  }
}

    
