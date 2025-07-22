// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "barney/MPIContext.h"
#include "barney/globalTrace/All2all.h"
#include "barney/DeviceGroup.h"
#include "barney/render/RayQueue.h"
#include "rtcore/ComputeInterface.h"

namespace BARNEY_NS {

  __rtc_global
  void createRayOnly(const rtc::ComputeInterface &ci,
                     RayOnly *rayOnly,
                     Ray *rayQueue,
                     int N)
  {
    int tid = ci.launchIndex().x;
    if (tid >= N) return;
    
    rayOnly[tid].org = rayQueue[tid].org;
    rayOnly[tid].dir = rayQueue[tid].dir;
    rayOnly[tid].tMax = rayQueue[tid].tMax;
    rayOnly[tid].isInMedium = rayQueue[tid].isInMedium;
    rayOnly[tid].isSpecular = rayQueue[tid].isSpecular;
    rayOnly[tid].isShadowRay = rayQueue[tid].isShadowRay;
  }

  __rtc_global
  void buildHitsOnly(const rtc::ComputeInterface &ci,
                     HitOnly *hitOnly,
                     Ray *rayQueue,
                     int N)
  {
    int tid = ci.launchIndex().x;
    if (tid >= N) return;

    hitOnly[tid].tHit = rayQueue[tid].tMax;
    hitOnly[tid].P = rayQueue[tid].P;
    hitOnly[tid].N = rayQueue[tid].N;
    hitOnly[tid].hitBSDF = rayQueue[tid].hitBSDF;
    hitOnly[tid].bsdfType = rayQueue[tid].bsdfType;
  }


  
  /*! given the set of rays we've recevied across all the differnet
      peers, reformat them into the classical 'barney::render::Ray'
      format, and put them into the rayqueue that the tracekernel
      would expect */
  __rtc_global
  void buildStagedRayQueue(const rtc::ComputeInterface &ci,
                           Ray *rayQueue,
                           RayOnly *rayOnly,
                           int N)
  {
    int tid = ci.launchIndex().x;
    if (tid >= N) return;

    rayQueue[tid].org = rayOnly[tid].org;
    rayQueue[tid].dir = rayOnly[tid].dir;
    rayQueue[tid].tMax = rayOnly[tid].tMax;
    rayQueue[tid].isInMedium = rayOnly[tid].isInMedium;
    rayQueue[tid].isSpecular = rayOnly[tid].isSpecular;
    rayQueue[tid].isShadowRay = rayOnly[tid].isShadowRay;
    rayQueue[tid].bsdfType = PackedBSDF::NONE; 
  }
  
  __rtc_global
  void reduceReceivedHits(const rtc::ComputeInterface &ci,
                          Ray *rayQueueThisRank,
                          HitOnly *hitOnlyAllRanks,
                          int nRays,
                          int islandSize)
  {
    int tid = ci.launchIndex().x;
    if (tid >= nRays) return;

    Ray ray = rayQueueThisRank[tid];
    for (int peer=0;peer<islandSize;peer++) {
      HitOnly *hit = hitOnlyAllRanks+peer*nRays+tid;

      if (hit->tHit >= ray.tMax) continue;

      ray.tMax     = hit->tHit;
      ray.bsdfType = hit->bsdfType;
      ray.hitBSDF  =  hit->hitBSDF;
      ray.P        =  hit->P;
      ray.N        =  hit->N;
    }
    rayQueueThisRank[tid] = ray;
  }


  MPIAll2all::MPIAll2all(MPIContext *context)
    : GlobalTraceImpl(context),
      context(context)
  {
    perLogical.resize(context->devices->numLogical);
  }
  
  MPIAll2all::PLD *MPIAll2all::getPLD(Device *device)
  {
    assert(device);
    assert(device->contextRank() >= 0);
    assert(device->contextRank() < perLogical.size());
    return &perLogical[device->contextRank()];
  }

  void MPIAll2all::ensureAllOurQueuesAreLargeEnough()
  {
    for (auto device : *context->devices) {
      auto rtc = device->rtc;
      PLD *pld = getPLD(device);
      size_t ourRequiredQueueSize
        = device->rayQueue->size * context->topo->islandSize();
      if (ourRequiredQueueSize > pld->currentSize) {
        std::cout << "resizing ray queues from " << pld->currentSize << " to " << ourRequiredQueueSize << std::endl;
        if (pld->send.raysOnly) rtc->freeMem(pld->send.raysOnly);
        if (pld->recv.raysOnly) rtc->freeMem(pld->recv.raysOnly);
        if (pld->send.hitsOnly) rtc->freeMem(pld->send.hitsOnly);
        if (pld->recv.hitsOnly) rtc->freeMem(pld->recv.hitsOnly);

        if (pld->stagedRayQueue) rtc->freeMem(pld->stagedRayQueue);
        
        size_t N = ourRequiredQueueSize;
        pld->send.raysOnly = (RayOnly*)rtc->allocMem(N*sizeof(RayOnly));
        pld->recv.raysOnly = (RayOnly*)rtc->allocMem(N*sizeof(RayOnly));
        pld->send.hitsOnly = (HitOnly*)rtc->allocMem(N*sizeof(HitOnly));
        pld->recv.hitsOnly = (HitOnly*)rtc->allocMem(N*sizeof(HitOnly));
        pld->stagedRayQueue = (Ray *)rtc->allocMem(N*sizeof(Ray));
        
        pld->currentSize = N;
      }
    }
  }

    // step 1: have all ranks exchange which (global) device has how
    // many rays (needed to set up the send/receives)
  void MPIAll2all::exchangeHowManyRaysEachDeviceHas()
  {
    auto &world = context->world;
    auto topo = context->topo;
    
    int islandSize = context->topo->islandSize();
    std::vector<MPI_Request> requests;
    for (auto device : *context->devices) {
      auto pld = getPLD(device);
      pld->perIslandPeer.rayCount.resize(islandSize);

      /* iw: use reference here, to make sure the send doesn't use a stack temp */
      int &myRayCount = device->rayQueue->numActive;
      const std::vector<int> &peers
        = topo->islands[topo->islandOf[device->globalRank()]];
      for (auto peer : peers) {
        auto &peerDev = topo->allDevices[peer];
        MPI_Request req;
        // world.send(peerDev.worldRank,peerDev.local,&myRayCount,1,req);
        // requests.push_back(req);
        world.recv(peerDev.worldRank,peerDev.local,
                   &pld->perIslandPeer.rayCount[topo->islandRankOf[peer]],1,req);
        requests.push_back(req);
      }
      for (auto peer : peers) {
        auto &peerDev = topo->allDevices[peer];
        MPI_Request req;
        world.send(peerDev.worldRank,peerDev.local,&myRayCount,1,req);
        requests.push_back(req);
        // world.recv(peerDev.worldRank,peerDev.local,
        //            &pld->perIslandPeer.rayCount[topo->islandRankOf[peer]],1,req);
        // requests.push_back(req);
      }
    }
    BN_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
    requests.clear();

#if 0
    for (auto device : *context->devices) {
      auto pld = getPLD(device);
      std::stringstream ss;
      ss << "#bn.a2a(" << device->worldRank() << "." << device->localRank() << "):";
      ss << " received ray counts";
      for (auto cnt : pld->perIslandPeer.rayCount)
        ss << " " << cnt;
      std::cout << ss.str() << std::endl;
    }
#endif
  }

  
  void MPIAll2all::traceAllReceivedRays(GlobalModel *model, uint32_t rngSeed, bool needHitIDs)
  {
    auto &world = context->world;
    auto topo = context->topo;

    for (auto device : *context->devices) {
      SetActiveGPU forDuration(device);
      PLD *pld = getPLD(device);
      __rtc_launch(device->rtc,
                   buildStagedRayQueue,
                   divRoundUp(pld->numRemoteRaysReceived,1024),1024,
                   // args
                   pld->stagedRayQueue,
                   pld->recv.raysOnly,
                   pld->numRemoteRaysReceived);

    }
    
    for (auto device : *context->devices) {
      PLD *pld = getPLD(device);
      device->rtc->sync();
      // receiveAndShadeWriteQueue
      pld->savedOriginalRayCount = device->rayQueue->numActive;
      pld->savedOriginalRayQueue = device->rayQueue->traceAndShadeReadQueue.rays;
      device->rayQueue->traceAndShadeReadQueue.rays = pld->stagedRayQueue;
      device->rayQueue->numActive = pld->numRemoteRaysReceived;
    }

    assert(needHitIDs == false); // not implemented right now
    context->traceRaysLocally(model,rngSeed,needHitIDs);
    // world.barrier(); sleep(context->myRank()); PING;
  }
  

  void MPIAll2all::sendAndReceiveRays()
  {
    auto &world = context->world;
    auto topo = context->topo;

    for (auto device : *context->devices) {
      SetActiveGPU forDuration(device);
      PLD *pld = getPLD(device);
      int numRays = device->rayQueue->numActive;
      __rtc_launch(device->rtc,
                   createRayOnly,
                   divRoundUp(numRays,128),128,
                   // args
                   pld->send.raysOnly,
                   device->rayQueue->traceAndShadeReadQueue.rays,
                   numRays);
    }
    
    std::vector<MPI_Request> requests;
    for (auto device : *context->devices) {
      device->rtc->sync();
      auto pld = getPLD(device);

      pld->numRemoteRaysReceived = 0;      
      int myRayCount = device->rayQueue->numActive;
      const std::vector<int> &peers
        = topo->islands[topo->islandOf[device->globalRank()]];
      int recvOfs = 0;
      for (auto peer : peers) {
        auto &peerDev = topo->allDevices[peer];
        int peerIslandRank = topo->islandRankOf[peer];
        int recvCount = pld->perIslandPeer.rayCount[peerIslandRank];
        if (peer == device->_globalRank) {
          device->rtc->copyAsync(pld->recv.raysOnly+recvOfs,
                                 pld->send.raysOnly,
                                 recvCount*sizeof(RayOnly));
        } else {
          MPI_Request req;
          world.recv(peerDev.worldRank,peerDev.local,
                     pld->recv.raysOnly+recvOfs,recvCount,req);
          requests.push_back(req);
        }
        recvOfs += recvCount;
        // world.send(peerDev.worldRank,peerDev.local,
        //            pld->send.raysOnly,myRayCount,req);
        // requests.push_back(req);
      }
      for (auto peer : peers) {
        auto &peerDev = topo->allDevices[peer];
        MPI_Request req;
        int peerIslandRank = topo->islandRankOf[peer];
        int recvCount = pld->perIslandPeer.rayCount[peerIslandRank];

        // world.recv(peerDev.worldRank,peerDev.local,
        //            pld->recv.raysOnly+recvOfs,recvCount,req);
        // requests.push_back(req);
        // recvOfs += recvCount;
        
        if (peer == device->_globalRank) {
        } else {
          world.send(peerDev.worldRank,peerDev.local,
                     pld->send.raysOnly,myRayCount,req);
          requests.push_back(req);
        }
      }
      pld->numRemoteRaysReceived = recvOfs;
    }
    BN_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
    requests.clear();
  }


  void MPIAll2all::sendAndReceiveHits()
  {
    auto &world = context->world;
    auto topo = context->topo;

    for (auto device : *context->devices) {
      device->rtc->sync();
      auto pld = getPLD(device);
      __rtc_launch(device->rtc,
                   buildHitsOnly,
                   divRoundUp(pld->numRemoteRaysReceived,1024),1024,
                   // args
                   pld->send.hitsOnly,
                   device->rayQueue->traceAndShadeReadQueue.rays,
                   pld->numRemoteRaysReceived);
    }
    
    std::vector<MPI_Request> requests;
    for (auto device : *context->devices) {
      device->rtc->sync();
      auto pld = getPLD(device);

      int myRayCount = pld->savedOriginalRayCount;
      const std::vector<int> &peers
        = topo->islands[topo->islandOf[device->globalRank()]];
      int sendOfs = 0;
      int recvOfs = 0;
      for (auto peer : peers) {
        auto &peerDev = topo->allDevices[peer];
        MPI_Request req;
        int peerIslandRank = topo->islandRankOf[peer];
        int sendCount = pld->perIslandPeer.rayCount[peerIslandRank];
        int recvCount = myRayCount;
        if (peer == device->globalRank()) {
          device->rtc->copyAsync(pld->recv.hitsOnly+recvOfs,
                                 pld->send.hitsOnly+sendOfs,
                                 recvCount*sizeof(HitOnly));
        } else {
          world.recv(peerDev.worldRank,peerDev.local,
                     pld->recv.hitsOnly+recvOfs,recvCount,req);
          requests.push_back(req);
        }
        recvOfs += recvCount;
        sendOfs += sendCount;
        
        // world.send(peerDev.worldRank,peerDev.local,
        //            pld->send.hitsOnly+sendOfs,sendCount,req);
        // requests.push_back(req);
        // sendOfs += sendCount;
      }
        sendOfs = 0;
      for (auto peer : peers) {
        auto &peerDev = topo->allDevices[peer];
        MPI_Request req;
        int peerIslandRank = topo->islandRankOf[peer];
        int sendCount = pld->perIslandPeer.rayCount[peerIslandRank];
        int recvCount = myRayCount;
        // world.recv(peerDev.worldRank,peerDev.local,
        //            pld->recv.hitsOnly+recvOfs,recvCount,req);
        // requests.push_back(req);
        // recvOfs += recvCount;

        if (peer == device->globalRank()) {
        } else {
          world.send(peerDev.worldRank,peerDev.local,
                     pld->send.hitsOnly+sendOfs,sendCount,req);
          requests.push_back(req);
        }
        sendOfs += sendCount;
      }
    }
    BN_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
    requests.clear();
  }

  // step 5: merge all the received hits back with the rays that
  // spawend them, and write them into local ray queue.
  void MPIAll2all::mergeReceivedHitsWithOriginalRays()
  {
    auto &world = context->world;
    auto topo = context->topo;
    int islandSize = context->topo->islandSize();

    for (auto device : *context->devices) {
      SetActiveGPU forDuration(device);
      PLD *pld = getPLD(device);
      int myRayCount = pld->savedOriginalRayCount;
      __rtc_launch(device->rtc,
                   reduceReceivedHits,
                   divRoundUp(myRayCount,128),128,
                   // args
                   pld->savedOriginalRayQueue,
                   pld->recv.hitsOnly,
                   myRayCount,
                   islandSize);
      device->rayQueue->numActive = myRayCount;
      device->rayQueue->traceAndShadeReadQueue.rays = pld->savedOriginalRayQueue;
    }

    for (auto device : *context->devices)
      device->rtc->sync();
  }

  
  void MPIAll2all::traceRays(GlobalModel *model, uint32_t rngSeed, bool needHitIDs)
  {
    double t0 = getCurrentTime();
    ensureAllOurQueuesAreLargeEnough();
    exchangeHowManyRaysEachDeviceHas();

    sendAndReceiveRays();
    traceAllReceivedRays(model,rngSeed,needHitIDs);
    
    sendAndReceiveHits();
    double t1 = getCurrentTime();
    mergeReceivedHitsWithOriginalRays();    

    if (context->myRank() == 0) PRINT(t1-t0);
  }
}
