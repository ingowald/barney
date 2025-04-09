// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

#include "barney/MPIContext.h"
#include "barney/fb/DistFB.h"
#include "barney/render/RayQueue.h"

#if 0
# define LOG_API_ENTRY std::cout << OWL_TERMINAL_BLUE << "#bn: " << __FUNCTION__ << OWL_TERMINAL_DEFAULT << std::endl;
#else
# define LOG_API_ENTRY /**/
#endif


#if defined(BARNEY_RTC_EMBREE) && defined(BARNEY_RTC_OPTIX)
# error "should not have both backends on at the same time!?"
#endif

namespace BARNEY_NS {

  MPIContext::MPIContext(const barney_api::mpi::Comm &worldComm,
                         const barney_api::mpi::Comm &workersComm,
                         bool isActiveWorker,
                         const std::vector<int> &dataGroupIDs,
                         const std::vector<int> &gpuIDs)
    : Context(dataGroupIDs,gpuIDs,
              isActiveWorker?workersComm.rank:0,
              isActiveWorker?workersComm.size:1),
      world(worldComm),
      workers(workersComm)
  {
    bool dbg = FromEnv::get()->logConfig;

    if (dbg) {
      std::stringstream ss;
      ss << "#bn." << workers.rank << " data group IDs ";
      for (auto dgID : dataGroupIDs)
        ss << dgID << " ";
      ss << std::endl;
      std::cout << ss.str();
    }
    world.assertValid();
    workers.assertValid();



    workerRankOfWorldRank.resize(world.size);
    world.allGather(workerRankOfWorldRank.data(),
                    isActiveWorker?workers.rank:-1);
    worldRankOfWorker.resize(workers.size);
    numWorkers = 0;
    for (int i=0;i<workerRankOfWorldRank.size();i++)
      if (workerRankOfWorldRank[i] != -1) {
        if (workerRankOfWorldRank[i] >= workers.size)
          throw std::runtime_error("Invalid worker rank!?");
        worldRankOfWorker[workerRankOfWorldRank[i]] = i;
        numWorkers++;
      }
    workers.size = numWorkers;

    gpusPerWorker = world.allReduceMax(int(gpuIDs.size()));
    numWorkers = world.allReduceAdd(isActiveWorker?1:0);

    if (dbg) {
      std::stringstream ss;
      ss << "bn." << workers.rank << ": ";
      ss << "num workers active/total " << numWorkers << "/" << workers.size << std::endl;
      std::cout << ss.str();
    }

    if (isActiveWorker) {
      int numSlotsPerWorker = (int)perSlot.size();
      int numDevicesPerWorker = contextSize();//(int)devices.size();
      int numWorkers = workers.size;

      int _globalID = workers.rank*numDevicesPerWorker;
      for (auto device : *devices) {
        device->globalRank = _globalID++;
        device->globalSize = numWorkers * numDevicesPerWorker;
      }
      
      if (dbg) {
        std::stringstream ss;
        ss << "bn." << workers.rank << ": ";
        ss << "num devices " << numDevicesPerWorker
           << " (";
        for (auto device : *devices)
          ss << " " << device->globalRank;
        ss << " ) DGs " << numSlotsPerWorker << std::endl << std::flush;
        std::cout << ss.str();
      }

      // ------------------------------------------------------------------
      // sanity check - make sure all workers have same num data groups
      // ------------------------------------------------------------------
      std::vector<int> numModelSlotsOnWorker(workers.size+1);
      numModelSlotsOnWorker[workers.size] = 0x290374;
      workers.allGather(numModelSlotsOnWorker.data(),numSlotsPerWorker);
      if (numModelSlotsOnWorker[workers.size] != 0x290374)
        throw std::runtime_error("mpi buffer overwrite!");
      for (int i=0;i<numWorkers;i++)
        if (numModelSlotsOnWorker[i] != numSlotsPerWorker)
          throw std::runtime_error
            ("worker rank "+std::to_string(i)+
             " has different number of data groups ("+
             std::to_string(numModelSlotsOnWorker[i])+
             " than worker rank "+std::to_string(workers.rank)+
             " ("+std::to_string(numSlotsPerWorker)+")");

      // ------------------------------------------------------------------
      // sanity check - make sure all workers have same num devices
      // ------------------------------------------------------------------
      std::vector<int> numDevicesOnWorker(workers.size+1);
      numDevicesOnWorker[workers.size] = 0x290375;
      workers.allGather(numDevicesOnWorker.data(),
                        devices->numLogical);//(int)devices.size());
      if (numDevicesOnWorker[workers.size] != 0x290375)
        throw std::runtime_error("mpi buffer overwrite!");
      for (int i=0;i<numWorkers;i++)
        if (numDevicesOnWorker[i] != devices->size())
          throw std::runtime_error
            ("worker rank "+std::to_string(i)+
             " has different number of data groups ("+
             std::to_string(numDevicesOnWorker[i])+
             " than worker rank "+std::to_string(workers.rank)+
             " ("+std::to_string(devices->size())+")");
      int numDevicesTotal = numDevicesOnWorker[0] * workers.size;

      // ------------------------------------------------------------------
      // gather who has which data(groups)
      // ------------------------------------------------------------------
      std::vector<int> allModelSlots(workers.size*numSlotsPerWorker+1);
      allModelSlots[workers.size*numSlotsPerWorker] = 0x8628;
      workers.allGather(allModelSlots.data(),
                        dataGroupIDs.data(),
                        dataGroupIDs.size());
      if (allModelSlots[workers.size*numSlotsPerWorker] != 0x8628)
        throw std::runtime_error("mpi buffer overwrite!");
      allModelSlots.resize(workers.size*numSlotsPerWorker);

      // ------------------------------------------------------------------
      // sanity check: data groups are numbered 0,1,2 .... and each
      // data group appears same number of times.
      // ------------------------------------------------------------------
      std::map<int,int> dataGroupCount;
      int maxModelSlotID = -1;
      for (int i=0;i<allModelSlots.size();i++) {
        int dgID_i = allModelSlots[i];
        if (dgID_i < 0)
          throw std::runtime_error
            ("invalid data group ID ("+std::to_string(dgID_i)+")");
        maxModelSlotID = std::max(maxModelSlotID,dgID_i);
        dataGroupCount[dgID_i]++;
      }
      numDifferentModelSlots = dataGroupCount.size();
      if (maxModelSlotID >= numDifferentModelSlots)
        throw std::runtime_error("data group IDs not numbered sequentially");

      int numIslands = dataGroupCount[0];
      for (auto dgc : dataGroupCount)
        if (dgc.second != numIslands)
          throw std::runtime_error
            ("some data groups used more often than others!?");

      // ------------------------------------------------------------------
      // for each local device, find which othe rdevice has 'next'
      // data group to cycle with. we already sanity checked that
      // there's symmetry in num devices, num data groups, etc.
      // ------------------------------------------------------------------
      std::vector<int> myDataOnLocal(devices->numLogical);
      for (auto slot : perSlot) 
        for (auto device : *slot.devices)
          myDataOnLocal[device->contextRank] = slot.modelRankInThisSlot;
      // for (int i=0;i<devices->size();i++)
      //   myDataOnLocal[i]
      //     = perSlot[(*devices)[i]->device->devGroup->lmsIdx].modelRankInThisSlot;
      if (dbg) {
        std::stringstream ss;
        ss << "bn." << workers.rank << ": ";
        ss << "*my* data ranks locally (myDataOnLocal): ";
        for (auto d : myDataOnLocal) ss << d << " ";
        std::cout << ss.str() << std::endl;
      }
      int numDevicesGlobally = numDevicesPerWorker*workers.size;
      std::vector<int> dataOnGlobal(numDevicesGlobally+1);
      dataOnGlobal[numDevicesGlobally] = 0x3723;
      workers.allGather(dataOnGlobal.data(),
                        myDataOnLocal.data(),
                        myDataOnLocal.size());
      if (dataOnGlobal[numDevicesGlobally] != 0x3723)
        throw std::runtime_error("mpi gather overrun");
      dataOnGlobal.resize(numDevicesGlobally);
      if (dbg) {
        std::stringstream ss;
        ss << "bn." << workers.rank << ": ";
        ss << "*all* data ranks globally  (dataOnGlobal): ";
        for (auto d : dataOnGlobal) ss << d << " ";
        std::cout << ss.str() << std::endl;
      }

      dataGroupCount.clear();
      std::vector<int> islandOfGlobal(numDevicesGlobally);
      for (int i=0;i<numDevicesGlobally;i++) {
        islandOfGlobal[i]
          = dataGroupCount[dataOnGlobal[i]]++;
      }
      if (dbg) {
        std::stringstream ss;
        ss << "bn." << workers.rank << ": ";
        ss << "islandranks globally: ";
        for (auto d : islandOfGlobal) ss << d << " ";
        std::cout << ss.str() << std::endl;
      }

      for (auto &slot : perSlot) {
        for (auto device : *devices) {
          int localID  = device->contextRank;
          int myGlobal = device->globalRank;
          int myDG     = slot.modelRankInThisSlot;//dataOnGlobal[myGlobal];
          int myIsland = islandOfGlobal[myGlobal];
          int nextDG   = (myDG+1) % numDifferentModelSlots;
          int prevDG   = (myDG+numDifferentModelSlots-1) % numDifferentModelSlots;
          std::stringstream ss;
          ss << "#bn " << myRank() << "." << localID << " (=" << myGlobal
             << ") looking for DG link "
             << prevDG << "->" << myDG << "->" << nextDG
             << " (for island " << myIsland << ")" << std::endl;
          for (int peerGlobal=0;peerGlobal<numDevicesGlobally;peerGlobal++) {
            ss << " peer " << peerGlobal << " has island " << islandOfGlobal[peerGlobal] << " dg " << dataOnGlobal[peerGlobal] << std::endl;
            if (islandOfGlobal[peerGlobal] != myIsland)
              continue;
            if (dataOnGlobal[peerGlobal] == nextDG) {
              // *found* the global next
              device->rqs.recvWorkerRank  = peerGlobal / numDevicesPerWorker;
              device->rqs.recvWorkerLocal = peerGlobal % numDevicesPerWorker;
              ss << " FOUND next " << device->rqs.recvWorkerRank << "." << device->rqs.recvWorkerLocal << std::endl;
            }
            if (dataOnGlobal[peerGlobal] == prevDG) {
              // *found* the global prev
              device->rqs.sendWorkerRank  = peerGlobal / numDevicesPerWorker;
              device->rqs.sendWorkerLocal = peerGlobal % numDevicesPerWorker;
              ss << " FOUND prev " << device->rqs.sendWorkerRank << "." << device->rqs.sendWorkerLocal << std::endl;
            }
          }
          if (dbg)
            std::cout << ss.str();
            std::cout << "local device " << localID << " recvs from device " << device->rqs.recvWorkerRank << "." << device->rqs.recvWorkerLocal << ", and sends to " <<
              device->rqs.sendWorkerRank << "." << device->rqs.recvWorkerLocal << std::endl;
        }
      }
    }
    barrier(false);
  }

  /*! create a frame buffer object suitable to this context */
  std::shared_ptr<barney_api::FrameBuffer>
  MPIContext::createFrameBuffer(int owningRank)
  {
    return std::make_shared<DistFB>(this,devices,owningRank);
  }

  /*! returns how many rays are active in all ray queues, across all
    devices and, where applicable, across all ranks */
  int MPIContext::numRaysActiveGlobally()
  {
    assert(isActiveWorker);
    return workers.allReduceAdd(numRaysActiveLocally());
  }

  
  void MPIContext::render(Renderer    *renderer,
                          GlobalModel *model,
                          Camera      *camera,
                          FrameBuffer *_fb)
  {
    DistFB *fb = (DistFB *)_fb;
    if (isActiveWorker) {
      renderTiles(renderer,model,camera,fb);
      finalizeTiles(fb);
    }
    // ------------------------------------------------------------------
    // done rendering, let the frame buffer know about it, so it can
    // do whatever needs doing with the latest finalized tiles
    // ------------------------------------------------------------------
    fb->finalizeFrame();
  }

  /*! forward rays (during global trace); returns if _after_ that
    forward the rays need more tracing (true) or whether they're
    done (false) */
  bool MPIContext::forwardRays(bool needHitIDs)
  {
    int numDevices = devices->size();
    std::vector<MPI_Request> allRequests;

    if (FromEnv::get()->logQueues) 
      std::cout << "----- forwardRays (numSLots = "
                << numDifferentModelSlots << ")"
                << " -----------" << std::endl;
    
    syncCheckAll();
    if (numDifferentModelSlots == 1) {
      // do NOT copy or swap. rays are in trace queue, which is also
      // the shade read queue, so nothing to do.
      //
      // no more trace rounds required: return false
      return false;
    }

    // ------------------------------------------------------------------
    // exchange how many we're going to send/recv
    // ------------------------------------------------------------------
    std::vector<int> numIncoming(numDevices);
    std::vector<int> numOutgoing(numDevices);
    for (auto &ni : numIncoming) ni = -1;
    for (auto device : *devices) {
      auto &rays = *device->rayQueue;

      MPI_Request sendReq, recvReq;
      numOutgoing[device->contextRank] = device->rayQueue->numActive;

      if (FromEnv::get()->logQueues) {
        std::stringstream ss;
        ss << "#" << myRank() << "." << device->contextRank << ":" << std::endl;
        ss << "  sends " << numOutgoing[device->contextRank] << " to "
           <<  device->rqs.sendWorkerRank << "."
           << device->rqs.sendWorkerLocal << std::endl;
        ss << "  recvs from " << device->rqs.recvWorkerRank << "." << device->rqs.recvWorkerLocal << std::endl;
        std::cout << ss.str();
      }
      
      workers.recv(device->rqs.recvWorkerRank,
                   device->rqs.recvWorkerLocal,
                   &numIncoming[device->contextRank],1,recvReq);
      workers.send(device->rqs.sendWorkerRank,
                   device->rqs.sendWorkerLocal,
                   &numOutgoing[device->contextRank],1,sendReq);
      allRequests.push_back(sendReq);
      allRequests.push_back(recvReq);
    }

    // allStatuses.resize(allRequests.size());
    // BN_MPI_CALL(Waitall(allRequests.size(),allRequests.data(),allStatuses.data()));
    if (FromEnv::get()->logQueues)
      std::cout << "before waitall" << std::endl;
    BN_MPI_CALL(Waitall(allRequests.size(),allRequests.data(),MPI_STATUSES_IGNORE));
    if (FromEnv::get()->logQueues)
      std::cout << "after waitall" << std::endl;
    
    // barrier(false);
    // for (int i=0;i<allStatuses.size();i++) {
    //   auto &status = allStatuses[i];
    //   if (status.MPI_ERROR != MPI_SUCCESS)
    //     throw std::runtime_error("error in mpi send/recv status!?");
    // }
    allRequests.clear();
    // allStatuses.clear();

    // ------------------------------------------------------------------
    // exchange actual rays
    // ------------------------------------------------------------------
    for (auto device : *devices) {
      numOutgoing[device->contextRank] = device->rayQueue->numActive;
      if (FromEnv::get()->logQueues)
        std::cout << myRank() << ": numOutgoing[" << device->contextRank
                  << "] = " << device->rayQueue->numActive << std::endl;
      MPI_Request sendReq, recvReq;
      workers.recv(device->rqs.recvWorkerRank,
                   device->rqs.recvWorkerLocal,
                   device->rayQueue->receiveAndShadeWriteQueue.rays,
                   numIncoming[device->contextRank],
                   recvReq);
      workers.send(device->rqs.sendWorkerRank,
                   device->rqs.sendWorkerLocal,
                   device->rayQueue->traceAndShadeReadQueue.rays,
                   numOutgoing[device->contextRank],
                   sendReq);
      allRequests.push_back(sendReq);
      allRequests.push_back(recvReq);
      if (needHitIDs) {
        workers.recv(device->rqs.recvWorkerRank,
                     device->rqs.recvWorkerLocal,
                     device->rayQueue->receiveAndShadeWriteQueue.hitIDs,
                     numIncoming[device->contextRank],
                     recvReq);
        workers.send(device->rqs.sendWorkerRank,
                     device->rqs.sendWorkerLocal,
                     device->rayQueue->traceAndShadeReadQueue.hitIDs,
                     numOutgoing[device->contextRank],
                     sendReq);
        allRequests.push_back(sendReq);
        allRequests.push_back(recvReq);
      }
    }
    // allStatuses.resize(allRequests.size());
    // BN_MPI_CALL(Waitall(allRequests.size(),allRequests.data(),allStatuses.data()));
    if (FromEnv::get()->logQueues)
      std::cout << "2nd waitall" << std::endl;
    BN_MPI_CALL(Waitall(allRequests.size(),allRequests.data(),MPI_STATUSES_IGNORE));
    if (FromEnv::get()->logQueues)
      std::cout << "after 2nd waitall" << std::endl;
    // barrier(false);
    // for (int i=0;i<allStatuses.size();i++) {
    //   auto &status = allStatuses[i];
    //   if (status.MPI_ERROR != MPI_SUCCESS)
    //     throw std::runtime_error("error in mpi send/recv status!?");
    // }
    allRequests.clear();
    // allStatuses.clear();

    // ------------------------------------------------------------------
    // now all rays should be exchanged -- swap queues
    // ------------------------------------------------------------------
    for (auto device : *devices) {
      device->rayQueue->swapAfterCycle(numTimesForwarded  % numDifferentModelSlots,
                                       numDifferentModelSlots);
      device->rayQueue->numActive = numIncoming[device->contextRank];
    }

    ++numTimesForwarded;
    return (numTimesForwarded % numDifferentModelSlots) != 0;
  }

  extern "C" {
# if BARNEY_RTC_EMBREE
    barney_api::Context *
    createMPIContext_embree(barney_api::mpi::Comm world,
                            barney_api::mpi::Comm workers,
                            bool isActiveWorker,
                            const std::vector<int> &dgIDs)
    {
      std::vector<int> gpuIDs = { 0 }; 
      return new BARNEY_NS::MPIContext(world,workers,isActiveWorker,
                                       dgIDs,gpuIDs);
    }
# endif
# if BARNEY_RTC_OPTIX
    barney_api::Context *
    createMPIContext_optix(barney_api::mpi::Comm world,
                           barney_api::mpi::Comm workers,
                           bool isActiveWorker,
                           const std::vector<int> &dgIDs,
                           const std::vector<int> &gpuIDs)
    {
      return new BARNEY_NS::MPIContext(world,workers,isActiveWorker,
                                       dgIDs,gpuIDs);
    }
# endif
  }
}
