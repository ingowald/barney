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

#include "barney/render/RayQueue.h"
#include "barney/Context.h"

namespace BARNEY_NS {

  RayQueue::RayQueue(Device *device)
    : device(device)
  {
    auto rtc = device->rtc;
    h_numActive = (int*)rtc->allocHost(sizeof(int));
  }
    
  RayQueue::~RayQueue()
  {
    auto rtc = device->rtc;
    rtc->freeHost(h_numActive);
  }

  void SingleQueue::alloc(rtc::Device *rtc, int size)
  {
    rays = (Ray *)rtc->allocMem(size*sizeof(Ray));
    states = (PathState *)rtc->allocMem(size*sizeof(PathState));
    hitIDs = (HitIDs *)rtc->allocMem(size*sizeof(HitIDs));
  }
  
  void SingleQueue::free(rtc::Device *rtc)
  {
    if (rays) rtc->freeMem(rays);
    if (hitIDs) rtc->freeMem(hitIDs);
    if (states) rtc->freeMem(states);
    rays   = 0;
    hitIDs = 0;
    states = 0;
  }
  
  int RayQueue::readNumActive()
  {
    SetActiveGPU forDuration(device);
    auto rtc = device->rtc;
    rtc->copyAsync(h_numActive,_d_nextWritePos,sizeof(int));
    rtc->sync();
    numActive = *h_numActive;
    if (FromEnv::get()->logQueues)
      printf("#bn: ## ray queue read numactive %i\n",numActive);
    return *h_numActive;
  }
    
  /*! how many rays are active in the *READ* queue */
  int RayQueue::numActiveRays() const
  {
    return numActive;
  }
    
  void RayQueue::resetWriteQueue()
  {
    SetActiveGPU forDuration(device);
    auto rtc = device->rtc;
    rtc->memsetAsync(_d_nextWritePos,0,sizeof(int));
    rtc->sync();
  }
    
  void RayQueue::swapAfterGeneration()
  {
    if (FromEnv::get()->logQueues)
      printf("#bn: ## ray queue swap (after generation)\n");
    std::swap(receiveAndShadeWriteQueue.rays, traceAndShadeReadQueue.rays);
    std::swap(receiveAndShadeWriteQueue.states, traceAndShadeReadQueue.states);
    std::swap(receiveAndShadeWriteQueue.hitIDs, traceAndShadeReadQueue.hitIDs);
  }

  void RayQueue::swapAfterCycle(int cycleID, int numCycles)
  {
    if (FromEnv::get()->logQueues)
      printf("#bn: ## ray queue swap after cycle (cycle %i/%i)\n",cycleID,numCycles);
    std::swap(receiveAndShadeWriteQueue.rays, traceAndShadeReadQueue.rays);
    std::swap(receiveAndShadeWriteQueue.hitIDs, traceAndShadeReadQueue.hitIDs);
  }
  void RayQueue::swapAfterShade()
  {
    if (FromEnv::get()->logQueues)
      printf("#bn: ## ray queue swap after cycle (after shade)\n");
    std::swap(receiveAndShadeWriteQueue.rays, traceAndShadeReadQueue.rays);
    std::swap(receiveAndShadeWriteQueue.states, traceAndShadeReadQueue.states);
    std::swap(receiveAndShadeWriteQueue.hitIDs, traceAndShadeReadQueue.hitIDs);
  }
  
  void RayQueue::resize(int newSize
#if SINGLE_CYCLE_RQS
                        , int maxRaysAcrossAllRanks
#endif
                        )
  {
    SetActiveGPU forDuration(device);
    auto rtc = device->rtc;
    traceAndShadeReadQueue.free(rtc);
    receiveAndShadeWriteQueue.free(rtc);

#if SINGLE_CYCLE_RQS
    if (rqs.raysOnly) rtc->freeMem(rqs.raysOnly);
    if (rqs.hitsOnly) rtc->freeMem(rqs.hitsOnly);
    rqs.raysOnly = (RayOnly*)rtc->allocMem(maxRaysAcrossAllRanks*sizeof(RayOnly));
    rqs.hitsOnly = (HitOnly*)rtc->allocMem(maxRaysAcrossAllRanks*sizeof(HitOnly));
#endif
    
    // if (traceAndShadeReadQueue.rays) 
    //   rtc->freeMem(traceAndShadeReadQueue);
    // if (receiveAndShadeWriteQueue)
    //   rtc->freeMem(receiveAndShadeWriteQueue);

    if (!_d_nextWritePos) 
      _d_nextWritePos = (int*)rtc->allocMem(sizeof(int)); 

    // traceAndShadeReadQueue = (Ray*)rtc->allocMem(newSize*sizeof(Ray));
    // receiveAndShadeWriteQueue = (Ray*)rtc->allocMem(newSize*sizeof(Ray));
    traceAndShadeReadQueue.alloc(rtc,newSize);
    receiveAndShadeWriteQueue.alloc(rtc,newSize);
        
    size = newSize;

    resetWriteQueue();
  }
}

    
