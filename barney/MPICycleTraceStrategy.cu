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

#include "barney/MPICycleTraceStrategy.h"
#include "barney/render/RayQueue.h"
#include "barney/MPIContext.h"

namespace BARNEY_NS {

  MPICycleTraceStrategy::MPICycleTraceStrategy(MPIContext *context)
      : RayQueueCycleTraceStrategy(context),
        context(context)
    {}
  
  /*! forward rays (during global trace); returns if _after_ that
    forward the rays need more tracing (true) or whether they're
    done (false) */
  bool MPICycleTraceStrategy::forwardRays(bool needHitIDs)
  {
    auto topo = context->topo; assert(topo);
    auto &workers = context->workers;
    int numDevices = context->devices->size();
    std::vector<MPI_Request> allRequests;

    if (FromEnv::get()->logQueues) 
      std::cout << "----- forwardRays (islandSize = "
                << topo->islandSize() << ")"
                << " -----------" << std::endl;
    
    context->syncCheckAll();
    if (topo->islandSize() == 1) {
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
    for (auto device : *context->devices) {
      const PLD &pld = *getPLD(device);
      int sendWorkerRank  = pld.sendPartner->worker;
      int sendWorkerLocal = pld.sendPartner->local;
      int recvWorkerRank  = pld.recvPartner->worker;
      int recvWorkerLocal = pld.recvPartner->local;

      auto &rays = *device->rayQueue;

      MPI_Request sendReq, recvReq;
      numOutgoing[device->localRank()] = device->rayQueue->numActive;


      if (FromEnv::get()->logQueues) {
        std::stringstream ss;
        ss << "#" << context->myRank() << "." << device->localRank() << ":" << std::endl;
        ss << "  sends " << numOutgoing[device->localRank()] << " to "
           << sendWorkerRank << "." << sendWorkerLocal << std::endl;
        ss << "  recvs from "
           << recvWorkerRank << "." << recvWorkerLocal << std::endl;
        std::cout << ss.str();
      }
      
      workers.recv(recvWorkerRank,
                   recvWorkerLocal,
                   &numIncoming[device->localRank()],1,recvReq);
      workers.send(sendWorkerRank,
                   sendWorkerLocal,
                   &numOutgoing[device->localRank()],1,sendReq);
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
    for (auto device : *context->devices) {
      const PLD &pld = *getPLD(device);
      int sendWorkerRank  = pld.sendPartner->worker;
      int sendWorkerLocal = pld.sendPartner->local;
      int recvWorkerRank  = pld.recvPartner->worker;
      int recvWorkerLocal = pld.recvPartner->local;

      numOutgoing[device->localRank()] = device->rayQueue->numActive;
      if (FromEnv::get()->logQueues)
        std::cout << context->myRank() << ": numOutgoing[" << device->localRank()
                  << "] = " << device->rayQueue->numActive << std::endl;
      MPI_Request sendReq, recvReq;
      workers.recv(recvWorkerRank,
                   recvWorkerLocal,
                   device->rayQueue->receiveAndShadeWriteQueue.rays,
                   numIncoming[device->localRank()],
                   recvReq);
      workers.send(sendWorkerRank,
                   sendWorkerLocal,
                   device->rayQueue->traceAndShadeReadQueue.rays,
                   numOutgoing[device->localRank()],
                   sendReq);
      allRequests.push_back(sendReq);
      allRequests.push_back(recvReq);
      if (needHitIDs) {
        workers.recv(recvWorkerRank,
                     recvWorkerLocal,
                     device->rayQueue->receiveAndShadeWriteQueue.hitIDs,
                     numIncoming[device->localRank()],
                     recvReq);
        workers.send(sendWorkerRank,
                     sendWorkerLocal,
                     device->rayQueue->traceAndShadeReadQueue.hitIDs,
                     numOutgoing[device->localRank()],
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
    for (auto device : *context->devices) {
      device->rayQueue->swapAfterCycle(numTimesForwarded  % topo->islandSize(),
                                       topo->islandSize());
      device->rayQueue->numActive = numIncoming[device->localRank()];
    }

    ++numTimesForwarded;
    return (numTimesForwarded % topo->islandSize()) != 0;
  }

}

