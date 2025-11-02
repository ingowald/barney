// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/render/RayQueue.h"
#include "barney/Context.h"

namespace BARNEY_NS {

  RayQueue::RayQueue(Device *device)
    : device(device)
  {
    auto rtc = device->rtc;
    h_numActive = (int*)rtc->allocHost(sizeof(int));

    // resize(rayQueueSize);
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
      printf("#bn(%i): ## ray queue swap (after generation)\n",
             device->globalRank());
    std::swap(receiveAndShadeWriteQueue.rays, traceAndShadeReadQueue.rays);
    std::swap(receiveAndShadeWriteQueue.states, traceAndShadeReadQueue.states);
    std::swap(receiveAndShadeWriteQueue.hitIDs, traceAndShadeReadQueue.hitIDs);
  }

  void RayQueue::swapAfterCycle(int cycleID, int numCycles)
  {
    if (FromEnv::get()->logQueues)
      printf("#bn(%i): ## ray queue swap after cycle (cycle %i/%i)\n",
             device->globalRank(),cycleID,numCycles);
    std::swap(receiveAndShadeWriteQueue.rays, traceAndShadeReadQueue.rays);
    std::swap(receiveAndShadeWriteQueue.hitIDs, traceAndShadeReadQueue.hitIDs);
  }
  void RayQueue::swapAfterShade()
  {
    if (FromEnv::get()->logQueues)
      printf("#bn(%i): ## ray queue swap after cycle (after shade)\n",
             device->globalRank());
    std::swap(receiveAndShadeWriteQueue.rays, traceAndShadeReadQueue.rays);
    std::swap(receiveAndShadeWriteQueue.states, traceAndShadeReadQueue.states);
    std::swap(receiveAndShadeWriteQueue.hitIDs, traceAndShadeReadQueue.hitIDs);
  }
  
  void RayQueue::resize(int newSize)
  {
    if (newSize <= size) return;
    
    SetActiveGPU forDuration(device);
    auto rtc = device->rtc;
    traceAndShadeReadQueue.free(rtc);
    receiveAndShadeWriteQueue.free(rtc);

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

    
