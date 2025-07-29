// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "barney/MPIContext.h"
#include "barney/globalTrace/TwoStage.h"
#include "barney/DeviceGroup.h"
#include "barney/render/RayQueue.h"
#include "rtcore/ComputeInterface.h"

namespace BARNEY_NS {

  __rtc_global
  void buildHitsOnly(const rtc::ComputeInterface &ci,
                     HitOnly *hitOnly,
                     Ray *rayQueue,
                     int N);
  
  __rtc_global
  void reduceReceivedHitsKernel_intraNode(const rtc::ComputeInterface &ci,
                                          HitOnly *hitOnly,
                                          int nRays,
                                          int reduceFactor)
  {
    int tid = ci.launchIndex().x;
    if (tid >= nRays) return;

    HitOnly reduced = hitOnly[tid];
    for (int peer=1;peer<reduceFactor;peer++) {
      HitOnly *hit = hitOnly+peer*nRays+tid;
      
      if (hit->tHit >= reduced.tHit) continue;
      
      reduced = *hit;
    }
    hitOnly[tid] = reduced;
  }
  
  __rtc_global
  void reduceReceivedHitsKernel_crossNodes(const rtc::ComputeInterface &ci,
                                           Ray *rayQueueThisRank,
                                           HitOnly *hitOnlyAllRanks,
                                           int nRays,
                                           int reduceFactor)
  {
    int tid = ci.launchIndex().x;
    if (tid >= nRays) return;

    Ray ray = rayQueueThisRank[tid];
    for (int peer=0;peer<reduceFactor;peer++) {
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
  
  
  __rtc_global
  void createRayOnly(const rtc::ComputeInterface &ci,
                     RayOnly *rayOnly,
                     Ray *rayQueue,
                     int N);
  __rtc_global
  void buildStagedRayQueue(const rtc::ComputeInterface &ci,
                           Ray *rayQueue,
                           RayOnly *rayOnly,
                           int N);
  
  TwoStage::TwoStage(MPIContext *context)
    : GlobalTraceImpl(context),
      context(context),
      world(context->world),
      topo(context->topo.get())
  {
    if (context->devices->size() != 1)
      throw std::runtime_error
        ("twostage all2all currently only works for one device per rank");
    this->device = context->devices->get(0);
    
    if (topo->islands.size() != 1)
      throw std::runtime_error
        ("twostage all2all currently only works for a single island");

    myGID = device->globalRank();
    numGlobal = topo->allDevices.size();
    rayCounts.resize(numGlobal);
    // sanity check that all physical nodes have same number of GPUs
    std::map<size_t,int> gpuCountInHost;
    int numHosts = 0;
    for (int gid=0; gid<context->topo->allDevices.size(); gid++) {
      auto &dev = context->topo->allDevices[gid];
      gpuCountInHost[dev.hostNameHash]++;
      numHosts = std::max(numHosts,topo->physicalHostIndexOf[gid]+1);
    }
    gpusPerHost = gpuCountInHost.begin()->second;
    for (auto count : gpuCountInHost)
      if (count.second != gpusPerHost)
        throw std::runtime_error
          ("twostage all2all currently requires same number of GPUs on all ranks");
    assert(numHosts * gpusPerHost == context->topo->allDevices.size());

#if 1
    this->hostIdx = topo->physicalHostIndexOf[myGID];
    this->gpuIdx = topo->physicalGpuIndexOf[myGID];
    _rankOf.resize(numGlobal);

    std::vector<int> gidOfRank(numGlobal);
    world.allGather(gidOfRank.data(),&myGID,1);
    for (int r=0;r<numGlobal;r++)
      _rankOf[gidOfRank[r]] = r;
#else
    this->hostIdx = myGID / gpusPerHost;
    this->gpuIdx = myGID % gpusPerHost;
    this->numHosts = numGlobal / gpusPerHost;
#endif

    if (FromEnv::get()->logTopo) {
      world.barrier();
      if (context->myRank() == 0) {
        std::cout << "=========== TwoStage All2all ===========" << std::endl;
        std::cout << "- num MPI ranks (w/ one gpu each) " << numGlobal << std::endl;
        std::cout << "- detected num physical hosts " << numHosts << std::endl;
        std::cout << "- detected num (active) GPUs per host " << gpusPerHost << std::endl;
        for (int h=0;h<numHosts;h++)
          for (int g=0;g<gpusPerHost;g++) {
            std::cout << "- gpu on rank " << (rankOf(h,g))
                      << " is logical h" << h << "g" << g << " {"
                      << topo->toString(rankOf(h,g)) << "}" << std::endl;
          }
      }
      world.barrier();
    }
  }


  void TwoStage::ensureAllOurQueuesAreLargeEnough()
  {
    auto rtc = device->rtc;
    size_t ourRequiredQueueSize
      = device->rayQueue->size * numGlobal;
    if (ourRequiredQueueSize > currentReservedSize) {
      if (FromEnv::get()->logQueues) {
        std::cout << "resizing ray queues from " << currentReservedSize
                  << " to " << ourRequiredQueueSize << std::endl;
      }
      for (int i=0;i<2;i++)
        if (raysOnly[i]) rtc->freeMem(raysOnly[i]);
      for (int i=0;i<2;i++)
        if (hitsOnly[i]) rtc->freeMem(hitsOnly[i]);
      
      if (stagedRayQueue) rtc->freeMem(stagedRayQueue);
      
      size_t N = ourRequiredQueueSize;
      for (int i=0;i<2;i++)
        raysOnly[i] = (RayOnly*)rtc->allocMem(N*sizeof(RayOnly));
      for (int i=0;i<2;i++)
        hitsOnly[i] = (HitOnly*)rtc->allocMem(N*sizeof(HitOnly));
      stagedRayQueue = (Ray *)rtc->allocMem(N*sizeof(Ray));
      
      currentReservedSize = N;
    }
  }

  // step 1: have all ranks exchange which (global) device has how
  // many rays (needed to set up the send/receives)
  void TwoStage::exchangeHowManyRaysEachDeviceHas()
  {
    auto &world = context->world;

    int myRayCount = device->rayQueue->numActive;
    // world.barrier();
    // BN_MPI_CALL(Alltoall(/* sendbuf */&myRayCount,
    //                      /* OUR count */1,MPI_INT,
    //                      /*recvbuf*/rayCounts.data(),
    //                      1,MPI_INT,
    //                      world.comm));
    world.allGather(rayCounts.data(),&myRayCount,1);

    
    if (FromEnv::get()->logQueues)  {
      for (int i=0;i<numGlobal;i++) {
        world.barrier();
        if (myGID == i) {
          std::cout << "ray counts (" << rayCounts.size() << "):";
          for (auto rc : rayCounts) std::cout << " " << rc;
          std::cout << std::endl;
        }
        world.barrier();
      }
    }
  }
  
  
  /*! in this stage we have all each GPU exchange its rays with
    all GPUs that have same phsycail ID in all OTHER ranks, but NOT
    with other GPUs in same rank
  */
  void TwoStage::sendAndReceiveRays_crossNodes()
  {
    auto &world = context->world;
    
    // -----------------------------------------------------------------------------
    // first, create 'raysOnly[]' array, for each local device
    // -----------------------------------------------------------------------------
    int myRayCount = device->rayQueue->numActive;
    {
      SetActiveGPU forDuration(device);
      int bs = 128;
      int nb = divRoundUp(myRayCount,bs);
      __rtc_launch(device->rtc,
                   createRayOnly,
                   nb,bs,
                   // args
                   raysOnly[0],
                   device->rayQueue->traceAndShadeReadQueue.rays,
                   myRayCount);
    }

    device->rtc->sync();

    // world.barrier();
    std::vector<MPI_Request> requests;
    int recvOfs = 0;
    for (int h=0;h<numHosts;h++) {
      MPI_Request req;
      int recvCount = rayCounts[rankOf(h,gpuIdx)];
      if (FromEnv::get()->logQueues) 
        printf("splat-cross r%i receiving %i from %i (q 0->1)\n",
               myGID,recvCount,rankOf(h,gpuIdx));
      world.recv(rankOf(h,gpuIdx),0,raysOnly[1]+recvOfs,
                 recvCount,req);
      recvOfs += recvCount;
      requests.push_back(req);
    }
    intraNodes.numRaysReceived = recvOfs;
    if (FromEnv::get()->logQueues) 
      printf("splat-cross r%i total received %i\n",
             myGID,intraNodes.numRaysReceived);
    
    // world.barrier();
    for (int h=0;h<numHosts;h++) {
      MPI_Request req;
      if (FromEnv::get()->logQueues) 
        printf("splat-cross r%i sending %i to %i (q 0->1)\n",
               myGID,myRayCount,rankOf(h,gpuIdx));
      world.send(rankOf(h,gpuIdx),0,raysOnly[0],myRayCount,req);
      requests.push_back(req);
    }
    // world.barrier();
    
    BN_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
    requests.clear();
  }
  

  /*! in this stage we have all each GPU exchange its rays with
    all GPUs that have same phsycail ID in all OTHER ranks, but NOT
    with other GPUs in same rank
  */
  void TwoStage::sendAndReceiveRays_intraNode()
  {
    auto &world = context->world;
    // world.barrier();

    std::vector<MPI_Request> requests;
    int recvOfs = 0;
    for (int g=0;g<gpusPerHost;g++) {
      MPI_Request req;
      int raysOnPeer = 0;
      for (int h=0;h<numHosts;h++)
        raysOnPeer += rayCounts[rankOf(h,g)];
      if (FromEnv::get()->logQueues) 
        printf("splat-intra r%i receiving %i from %i (q 1->0)\n",
               myGID,raysOnPeer,rankOf(hostIdx,g));
      world.recv(rankOf(hostIdx,g),0,
                 raysOnly[0]+recvOfs,raysOnPeer,
                 req);
      recvOfs += raysOnPeer;
      requests.push_back(req);
    }
    bothStages.numRaysReceived = recvOfs;
    if (FromEnv::get()->logQueues) 
      printf("splat-intra r%i total received %i\n",
             myGID,bothStages.numRaysReceived);

    // world.barrier();
    
    int numRaysWeHave = 0;
    for (int h=0;h<numHosts;h++)
      numRaysWeHave += rayCounts[rankOf(h,gpuIdx)];
    for (int g=0;g<gpusPerHost;g++) {
      MPI_Request req;
      if (FromEnv::get()->logQueues) 
        printf("splat-intra r%i sending %i to %i (q 1->0)\n",
               myGID,numRaysWeHave,rankOf(hostIdx,g));
      world.send(rankOf(hostIdx,g),0,
                 raysOnly[1],numRaysWeHave,
                 req);
      requests.push_back(req);
    }

    // world.barrier();
    
    BN_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
    requests.clear();
  }

  

  void TwoStage::traceRays(GlobalModel *model,
                           uint32_t rngSeed,
                           bool needHitIDs) 
  {

    // std::cout << "==================================================================\n";
    // world.barrier();
    assert(needHitIDs == false); // not implemented right now
    // world.barrier();
    ensureAllOurQueuesAreLargeEnough();
    // world.barrier();
    exchangeHowManyRaysEachDeviceHas();
    // world.barrier();
    sendAndReceiveRays_crossNodes();
    // world.barrier();
    sendAndReceiveRays_intraNode();
    // world.barrier();

    traceAllReceivedRays(model,rngSeed,needHitIDs);
    // world.barrier();

    exchangeHits_intraNode();
     // world.barrier();
    reduceHits_intraNode();
    // world.barrier();
    exchangeHits_crossNodes();
    // world.barrier();
    reduceHits_crossNodes();
    // world.barrier();
  }


  void TwoStage::traceAllReceivedRays(GlobalModel *model,
                                      uint32_t rngSeed,
                                      bool needHitIDs)
  {
    SetActiveGPU forDuration(device);
    int numRaysWeHaveTotal = bothStages.numRaysReceived;
    if (FromEnv::get()->logQueues) 
      printf("buildlocalrays r%i total rays %i (q0)\n",
             myGID,numRaysWeHaveTotal);
    __rtc_launch(device->rtc,
                 buildStagedRayQueue,
                 divRoundUp(numRaysWeHaveTotal,1024),1024,
                 // args
                 stagedRayQueue,
                 raysOnly[0],
                 numRaysWeHaveTotal);
    
    device->rtc->sync();
    
    auto savedOriginalRayCount = device->rayQueue->numActive;
    auto savedOriginalRayQueue = device->rayQueue->traceAndShadeReadQueue.rays;
    device->rayQueue->traceAndShadeReadQueue.rays = stagedRayQueue;
    device->rayQueue->numActive = numRaysWeHaveTotal;

    if (FromEnv::get()->logQueues) 
      printf("localtrace r%i total rays %i\n",
             myGID,numRaysWeHaveTotal);
    context->traceRaysLocally(model,rngSeed,needHitIDs);
    device->rtc->sync();
    
    
    if (FromEnv::get()->logQueues) 
      printf("buildhits r%i total rays %i (q0)\n",
             myGID,numRaysWeHaveTotal);
    __rtc_launch(device->rtc,
                 buildHitsOnly,
                 divRoundUp(numRaysWeHaveTotal,1024),1024,
                 // args
                 hitsOnly[0],
                 stagedRayQueue,
                 numRaysWeHaveTotal);
    device->rtc->sync();
    
    device->rayQueue->numActive = savedOriginalRayCount;
    device->rayQueue->traceAndShadeReadQueue.rays = savedOriginalRayQueue;
  }
  
  void TwoStage::exchangeHits_intraNode()
  {
    std::vector<MPI_Request> requests;
    int recvOfs = 0;
    for (int g=0;g<gpusPerHost;g++) {
      MPI_Request req;
      int recvCount = 0;
      for (int h=0;h<numHosts;h++)
        recvCount += rayCounts[rankOf(h,gpuIdx)];
      world.recv(rankOf(hostIdx,g),0,
                 hitsOnly[1]+recvOfs,recvCount,req);
      if (FromEnv::get()->logQueues) 
        printf("xchg-intra r%i receiving %i from %i (q0->1)\n",
               myGID,recvCount,rankOf(hostIdx,g));
      requests.push_back(req);
      recvOfs += recvCount;
    }

    // and matching sends
    int sendOfs = 0;
    for (int g=0;g<gpusPerHost;g++) {
      int sendCount = 0;
      for (int h=0;h<numHosts;h++)
        sendCount += rayCounts[rankOf(h,g)];
      MPI_Request req;
      world.send(rankOf(hostIdx,g),0,
                 hitsOnly[0]+sendOfs,sendCount,req);
      if (FromEnv::get()->logQueues) 
        printf("xchg-intra r%i sending %i to %i (q0->1)\n",
               myGID,sendCount,rankOf(hostIdx,g));
      requests.push_back(req);
      sendOfs += sendCount;
    }
    BN_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
    requests.clear();
  }


  void TwoStage::reduceHits_intraNode()
  {
    SetActiveGPU forDuration(device);
    int g = gpuIdx;
    int numUniqueRaysThisGPU = 0;
    for (int h=0;h<numHosts;h++)
      numUniqueRaysThisGPU += rayCounts[rankOf(h,g)];

    if (FromEnv::get()->logQueues) 
      printf("r%i intra-reducing %i sets of %i hits (q1)\n",
             myGID,
             gpusPerHost,
             numUniqueRaysThisGPU);
    __rtc_launch(device->rtc,
                 reduceReceivedHitsKernel_intraNode,
                 divRoundUp(numUniqueRaysThisGPU,128),128,
                 // args
                 hitsOnly[1],
                 numUniqueRaysThisGPU,
                 gpusPerHost);
    device->rtc->sync();
  }
  
  void TwoStage::exchangeHits_crossNodes()
  {
    std::vector<MPI_Request> requests;
    int recvOfs = 0;
    int recvCount = rayCounts[rankOf(hostIdx,gpuIdx)];
    for (int h=0;h<numHosts;h++) {
      MPI_Request req;

      if (FromEnv::get()->logQueues) 
        printf("xchg-intra r%i receiving %i from %i (q1->0)\n",
               myGID,recvCount,rankOf(h,gpuIdx));
      world.recv(rankOf(h,gpuIdx),0,
                 hitsOnly[0]+recvOfs,recvCount,req);
      requests.push_back(req);
      recvOfs += recvCount;
    }

    // and matching sends
    int sendOfs = 0;
    for (int h=0;h<numHosts;h++) {
      MPI_Request req;
      int sendCount = rayCounts[rankOf(h,gpuIdx)];
      if (FromEnv::get()->logQueues) 
        printf("xchg-intra r%i sending %i to %i (q1->0)\n",
               myGID,sendCount,rankOf(h,gpuIdx));
      world.send(rankOf(h,gpuIdx),0,
                 hitsOnly[1]+sendOfs,sendCount,req);
      requests.push_back(req);
      sendOfs += sendCount;
    }
    
    BN_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
    requests.clear();
  }
  
  void TwoStage::reduceHits_crossNodes()
  {
    SetActiveGPU forDuration(device);
    int numUniqueRaysThisGPU = rayCounts[rankOf(hostIdx,gpuIdx)];
    if (FromEnv::get()->logQueues) 
      printf("r%i cross-reducing %i sets of %i hits (q0)\n",
             myGID,numHosts,numUniqueRaysThisGPU);
    __rtc_launch(device->rtc,
                 reduceReceivedHitsKernel_crossNodes,
                 divRoundUp(numUniqueRaysThisGPU,128),128,
                 // args
                 device->rayQueue->traceAndShadeReadQueue.rays,
                 hitsOnly[0],
                 numUniqueRaysThisGPU,
                 numHosts);
    device->rtc->sync();
  }
  
}
