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
#include "barney.h"

#if 0
# define LOG_API_ENTRY std::cout << OWL_TERMINAL_BLUE << "#bn: " << __FUNCTION__ << OWL_TERMINAL_DEFAULT << std::endl;
#else
# define LOG_API_ENTRY /**/
#endif

namespace barney {

  MPIContext::MPIContext(const mpi::Comm &worldComm,
                         const mpi::Comm &workersComm,
                         bool isActiveWorker,
                         const std::vector<int> &dataGroupIDs,
                         const std::vector<int> &gpuIDs)
    : Context(dataGroupIDs,gpuIDs,
              isActiveWorker?workersComm.rank:0,
              isActiveWorker?workersComm.size:1),
      world(worldComm),
      workers(workersComm)
  {
    bool dbg = true;

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
      ss << "num workers " << numWorkers << "/" << workers.size << std::endl;
    }

    if (isActiveWorker) {
      int numSlotsPerWorker = (int)perSlot.size();
      int numDevicesPerWorker = (int)devices.size();
      int numWorkers = workers.size;

      if (dbg) {
        std::stringstream ss;
        ss << "bn." << workers.rank << ": ";
        ss << "num devices " << numDevicesPerWorker << " DGs " << numSlotsPerWorker << std::endl << std::flush;
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
      workers.allGather(numDevicesOnWorker.data(),(int)devices.size());
      if (numDevicesOnWorker[workers.size] != 0x290375)
        throw std::runtime_error("mpi buffer overwrite!");
      for (int i=0;i<numWorkers;i++)
        if (numDevicesOnWorker[i] != devices.size())
          throw std::runtime_error
            ("worker rank "+std::to_string(i)+
             " has different number of data groups ("+
             std::to_string(numDevicesOnWorker[i])+
             " than worker rank "+std::to_string(workers.rank)+
             " ("+std::to_string(devices.size())+")");
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
      std::vector<int> myDataOnLocal(devices.size());
      for (int i=0;i<devices.size();i++)
        myDataOnLocal[i]
          = perSlot[devices[i]->device->devGroup->lmsIdx].modelRankInThisSlot;
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
        if (i >= islandOfGlobal.size())
          throw std::runtime_error("invalid islandofglobal access....");
        islandOfGlobal[i]
          = dataGroupCount[dataOnGlobal[i]]++;
      }

      for (int localID=0;localID<devices.size();localID++) {
        auto dev = devices[localID]->device;
        int myGlobal = dev->globalIndex;
        int myDG     = dataOnGlobal[myGlobal];
        int myIsland = islandOfGlobal[myGlobal];
        int nextDG   = (myDG+1) % numDifferentModelSlots;
        int prevDG   = (myDG+numDifferentModelSlots-1) % numDifferentModelSlots;
        for (int peerGlobal=0;peerGlobal<numDevicesGlobally;peerGlobal++) {
          if (islandOfGlobal[peerGlobal] != myIsland)
            continue;
          if (dataOnGlobal[peerGlobal] == nextDG) {
            // *found* the global next
            dev->rqs.recvWorkerRank  = peerGlobal / numDevicesPerWorker;
            dev->rqs.recvWorkerLocal = peerGlobal % numDevicesPerWorker;
          }
          if (dataOnGlobal[peerGlobal] == prevDG) {
            // *found* the global prev
            dev->rqs.sendWorkerRank  = peerGlobal / numDevicesPerWorker;
            dev->rqs.sendWorkerLocal = peerGlobal % numDevicesPerWorker;
          }
        }
        if (dbg)
          std::cout << "local device " << localID << " recvs from device " << dev->rqs.recvWorkerRank << "." << dev->rqs.recvWorkerLocal << ", and sends to " <<
            dev->rqs.sendWorkerRank << "." << dev->rqs.recvWorkerLocal << std::endl;
      }
    }
    barrier(false);
  }

  /*! create a frame buffer object suitable to this context */
  FrameBuffer *MPIContext::createFB(int owningRank)
  {
    return initReference(DistFB::create(this,owningRank));
  }

  /*! returns how many rays are active in all ray queues, across all
    devices and, where applicable, across all ranks */
  int MPIContext::numRaysActiveGlobally()
  {
    assert(isActiveWorker);
    return workers.allReduceAdd(numRaysActiveLocally());
  }


  void MPIContext::render(GlobalModel *model,
                          const Camera::DD &camera,
                          FrameBuffer *_fb,
                          int pathsPerPixel)
  {
    DistFB *fb = (DistFB *)_fb;
    if (isActiveWorker) {
      renderTiles(model,camera,fb,pathsPerPixel);
      finalizeTiles(fb);
    }
    // ------------------------------------------------------------------
    // done rendering, now gather all final tiles at master
    // ------------------------------------------------------------------
    fb->ownerGatherFinalTiles();

    // ==================================================================
    // now MASTER (who has gathered all the ranks' final tiles) -
    // writes them into proper row-major frame buffer order
    // (writeFinalPixels), then copies them to app FB). only master
    // can/shuld do this - ranks don't even have a 'finalFB' to
    // write into.
    // ==================================================================
    if (fb->isOwner) {
      /* ******************************************************* *
       CAREFUL: do NOT set active gpu here - the app might have its
       'finalFB' frame buffer allocated on another device than our
       device[0]; setting that to active will cause segfault when
       writing final pixel!!!!  *
       ******************************************************* */
      // SetActiveGPU forDuration(devices[0]->device);

      // use default gpu for this:
      barney::TiledFB::writeFinalPixels(// nullptr,
#if DENOISE
                                        fb->denoiserInput,
#else
                                        fb->finalFB,
#endif
                                        // fb->finalFB,
                                        fb->finalDepth,
# if DENOISE_NORMAL
                              fb->denoiserNormal,
# endif
                                        fb->numPixels,
                                        fb->ownerGather.finalTiles,
                                        fb->ownerGather.tileDescs,
                                        fb->ownerGather.numActiveTiles,
                                        fb->showCrosshairs);
#if DENOISE
      fb->denoise();
    // float4ToBGBA8(fb->finalFB,fb->denoiserInput,fb->numPixels);
#endif
      // copy to app framebuffer - only if we're the one having that
      // frame buffer of course
      BARNEY_CUDA_SYNC_CHECK();
      if (fb->hostFB && fb->finalFB != fb->hostFB) {
        BARNEY_CUDA_CALL(Memcpy(fb->hostFB,fb->finalFB,
                                fb->numPixels.x*fb->numPixels.y*sizeof(uint32_t),
                                cudaMemcpyDefault));
      }
      if (fb->hostDepth && fb->finalDepth != fb->hostDepth) {
        BARNEY_CUDA_CALL(Memcpy(fb->hostDepth,fb->finalDepth,
                                fb->numPixels.x*fb->numPixels.y*sizeof(float),
                                cudaMemcpyDefault));
      }
    }
    BARNEY_CUDA_SYNC_CHECK();
  }

  /*! forward rays (during global trace); returns if _after_ that
    forward the rays need more tracing (true) or whether they're
    done (false) */
  bool MPIContext::forwardRays()
  {
    if (numDifferentModelSlots == 1)
      return false;

    int numDevices = devices.size();
    std::vector<MPI_Request> allRequests;

    // ------------------------------------------------------------------
    // exchange how many we're going to send/recv
    // ------------------------------------------------------------------
    // std::vector<MPI_Status> allStatuses;
    std::vector<int> numIncoming(numDevices);
    std::vector<int> numOutgoing(numDevices);
    for (auto &ni : numIncoming) ni = -1;
    for (int devID=0;devID<numDevices;devID++) {
      auto dev = devices[devID]->device;
      auto &rays = devices[devID]->rays;

      MPI_Request sendReq, recvReq;
      numOutgoing[devID] = rays.numActive;
      workers.recv(dev->rqs.recvWorkerRank,dev->rqs.recvWorkerLocal,
                   &numIncoming[devID],1,recvReq);
      workers.send(dev->rqs.sendWorkerRank,
                   devID,//dev->rqs.sendWorkerLocal,
                   &numOutgoing[devID],1,sendReq);
      allRequests.push_back(sendReq);
      allRequests.push_back(recvReq);
    }

    // allStatuses.resize(allRequests.size());
    // BN_MPI_CALL(Waitall(allRequests.size(),allRequests.data(),allStatuses.data()));
     BN_MPI_CALL(Waitall(allRequests.size(),allRequests.data(),MPI_STATUSES_IGNORE));
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
    for (int devID=0;devID<numDevices;devID++) {
      auto dev = devices[devID]->device;
      auto &rays = devices[devID]->rays;

      numOutgoing[devID] = rays.numActive;
      MPI_Request sendReq, recvReq;
      workers.recv(dev->rqs.recvWorkerRank,dev->rqs.recvWorkerLocal,
                   rays.receiveAndShadeWriteQueue,numIncoming[devID],recvReq);
      workers.send(dev->rqs.sendWorkerRank,devID,//dev->rqs.sendWorkerLocal,
                   rays.traceAndShadeReadQueue,numOutgoing[devID],sendReq);
      // workers.recv(dev->rqs.recvWorkerRank,dev->rqs.recvWorkerLocal,
      //              rays.writeQueue,numIncoming[devID],recvReq);
      // workers.send(dev->rqs.sendWorkerRank,devID,//dev->rqs.sendWorkerLocal,
      //              rays.readQueue,numOutgoing[devID],sendReq);
      allRequests.push_back(sendReq);
      allRequests.push_back(recvReq);
    }
    // allStatuses.resize(allRequests.size());
    // BN_MPI_CALL(Waitall(allRequests.size(),allRequests.data(),allStatuses.data()));
    BN_MPI_CALL(Waitall(allRequests.size(),allRequests.data(),MPI_STATUSES_IGNORE));
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
    for (int devID=0;devID<numDevices;devID++) {
      auto dev = devices[devID];
      dev->rays.swap();
      dev->rays.numActive = numIncoming[devID];
    }

    ++numTimesForwarded;
    return (numTimesForwarded % numDifferentModelSlots) != 0;
  }

  BN_API
  void  bnMPIQueryHardware(BNHardwareInfo *_hardware, MPI_Comm _comm)
  {
    LOG_API_ENTRY;

    assert(_hardware);
    BNHardwareInfo &hardware = *_hardware;

    assert(_comm != MPI_COMM_NULL);
    mpi::Comm comm(_comm);

    hardware.numRanks = comm.size;
    char hostName[MPI_MAX_PROCESSOR_NAME];
    memset(hostName,0,MPI_MAX_PROCESSOR_NAME);
    int hostNameLen = 0;
    BN_MPI_CALL(Get_processor_name(hostName,&hostNameLen));

    std::vector<char> recvBuf(MPI_MAX_PROCESSOR_NAME*comm.size);
    memset(recvBuf.data(),0,recvBuf.size());

    // ------------------------------------------------------------------
    // determine which (world) rank lived on which host, and assign
    // GPUSs
    // ------------------------------------------------------------------
    BN_MPI_CALL(Allgather(hostName,
                          MPI_MAX_PROCESSOR_NAME,MPI_CHAR,
                          recvBuf.data(),
                          /* PER rank size */MPI_MAX_PROCESSOR_NAME,MPI_CHAR,
                          comm.comm));
    std::vector<std::string>  hostNames;
    std::map<std::string,int> ranksOnHost;
    for (int i=0;i<comm.size;i++)  {
      std::string host_i = recvBuf.data()+i*MPI_MAX_PROCESSOR_NAME;
      hostNames.push_back(host_i);
      ranksOnHost[host_i] ++;
    }

    hardware.numRanksThisHost = ranksOnHost[hostName];
    hardware.numHosts         = ranksOnHost.size();

    // ------------------------------------------------------------------
    // count how many other ranks are already on this same node
    // ------------------------------------------------------------------
    BN_MPI_CALL(Barrier(comm.comm));
    int localRank = 0;
    for (int i=0;i<comm.rank;i++)
      if (hostNames[i] == hostName)
        localRank++;
    BN_MPI_CALL(Barrier(comm.comm));
    hardware.localRank = localRank;
    hardware.numRanksThisHost = ranksOnHost[hostName];

    // ------------------------------------------------------------------
    // assign a GPU to this rank
    // ------------------------------------------------------------------
    int numGPUsOnThisHost;
    cudaGetDeviceCount(&numGPUsOnThisHost);
    if (numGPUsOnThisHost == 0)
      throw std::runtime_error("no GPU on this rank!");
    hardware.numGPUsThisHost = numGPUsOnThisHost;
    hardware.numGPUsThisRank
      = comm.allReduceMin(std::max(hardware.numGPUsThisHost/
                                   hardware.numRanksThisHost,
                                   1));
    assert(hardware.numGPUsThisRank > 0);
  }

  BN_API
  BNContext bnMPIContextCreate(MPI_Comm _comm,
                               /*! how many data slots this context is to
                                 offer, and which part(s) of the
                                 distributed model data these slot(s)
                                 will hold */
                               const int *dataRanksOnThisContext,
                               int        numDataRanksOnThisContext,
                               /*! which gpu(s) to use for this
                                 process. default is to distribute
                                 node's GPUs equally over all ranks on
                                 that given node */
                               const int *_gpuIDs,
                               int  numGPUs
                               )
  {
    LOG_API_ENTRY;

    mpi::Comm world(_comm);

    if (world.size == 1) {
      std::cout << "#bn: MPIContextInit, but only one rank - using local context" << std::endl;
      return bnContextCreate(dataRanksOnThisContext,
                             numDataRanksOnThisContext == 0
                             ? 1 : numDataRanksOnThisContext,
                               /*! which gpu(s) to use for this
                                 process. default is to distribute
                                 node's GPUs equally over all ranks on
                                 that given node */
                             _gpuIDs,
                             numGPUs);
      // return (BNContext)new LocalContext(dataGroupIDs,
      //                                    gpuIDs);
    }

    // ------------------------------------------------------------------
    // create vector of data groups; if actual specified by user we
    // use those; otherwise we use IDs
    // [0,1,...numModelSlotsOnThisHost)
    // ------------------------------------------------------------------
    assert(/* data groups == 0 is allowed for passive nodes*/
           numDataRanksOnThisContext >= 0);
    std::vector<int> dataGroupIDs;
    int rank;
    MPI_Comm_rank(world, &rank);
    for (int i=0;i<numDataRanksOnThisContext;i++)
      dataGroupIDs.push_back
        (dataRanksOnThisContext
         ? dataRanksOnThisContext[i]
         : rank*numDataRanksOnThisContext+i);

    // ------------------------------------------------------------------
    // create list of GPUs to use for this rank. if specified by user
    // we use this; otherwise we use GPUs in order, split into groups
    // according to how many ranks there are on this host. Ie, if host
    // has four GPUs the first rank will take 0 and 1; and the second
    // one will take 2 and 3.
    // ------------------------------------------------------------------
    BNHardwareInfo hardware;
    bnMPIQueryHardware(&hardware,_comm);

    std::vector<int> gpuIDs;
    if (_gpuIDs) {
      for (int i=0;i<numGPUs;i++)
        gpuIDs.push_back(_gpuIDs[i]);
    } else {
      if (numGPUs < 1)
        numGPUs = hardware.numGPUsThisRank;
      for (int i=0;i<numGPUs;i++)
        gpuIDs.push_back((hardware.localRank*hardware.numGPUsThisRank
                          + i) % hardware.numGPUsThisHost);
    }

    // if (1) {
    //   std::stringstream ss;
    //   ss << "#bn." << world.rank << ": gpuIDs ";
    //   for (auto i : gpuIDs) ss << i << " ";
    //   ss << std::endl;
    //   ss << "localRank " << hardware.localRank << std::endl;
    //   ss << "numGPUsThisRank " << hardware.numGPUsThisRank << std::endl;
    //   ss << "numGPUsThisHost " << hardware.numGPUsThisHost << std::endl;
    //   std::cout << ss.str() << std::flush;
    //   world.barrier();
    //   exit(0);
    // }

    bool isActiveWorker = !dataGroupIDs.empty();
    mpi::Comm workers = world.split(isActiveWorker);

    return (BNContext)new MPIContext(world,
                                     workers,
                                     isActiveWorker,
                                     dataGroupIDs,
                                     gpuIDs);
  }
}
