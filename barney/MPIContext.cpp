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
#include "barney/globalTrace/RQSMPI.h"
#include "barney/globalTrace/All2all.h"
#include "barney/globalTrace/TwoStage.h"

#if 0
# define LOG_API_ENTRY std::cout << OWL_TERMINAL_BLUE << "#bn: " << __FUNCTION__ << OWL_TERMINAL_DEFAULT << std::endl;
#else
# define LOG_API_ENTRY /**/
#endif


#if defined(BARNEY_RTC_EMBREE) && defined(BARNEY_RTC_OPTIX)
# error "should not have both backends on at the same time!?"
#endif

namespace BARNEY_NS {

  size_t getHostNameHash();
  
  WorkerTopo::SP
  MPIContext::makeTopo(const barney_api::mpi::Comm &worldComm,
                       const barney_api::mpi::Comm &workerComm,
                       const std::vector<LocalSlot> &localSlots)
  {
    std::vector<WorkerTopo::Device> devices;
    for (auto ls : localSlots) {
      for (auto gpuID : ls.gpuIDs) {
        WorkerTopo::Device dev;
        dev.local = devices.size();
        dev.worker = workerComm.rank;
        dev.worldRank = worldComm.rank;
        dev.dataRank = ls.dataRank;
        dev.hostNameHash = getHostNameHash();
        dev.physicalDeviceHash = rtc::getPhysicalDeviceHash(gpuID);
        devices.push_back(dev);
      }
    }
    int myCount = devices.size();
    int allCount = worldComm.allReduceAdd(myCount);

    std::vector<WorkerTopo::Device> allDevices(allCount);
    worldComm.allGather(allDevices.data(),
                        devices.data(),
                        myCount,
                        sizeof(WorkerTopo::Device));
    int myOfs = 0;
    for (myOfs=0;myOfs<allDevices.size();myOfs++)
      if (allDevices[myOfs].worldRank == worldComm.rank)
        break;
      
    return std::make_shared<WorkerTopo>(allDevices,myOfs,myCount);
  }

  
  inline bool isPassiveNode(const std::vector<LocalSlot> &localSlots)
  { return localSlots.size() == 1 && localSlots[0].dataRank == -1; }
  
  MPIContext::MPIContext(const barney_api::mpi::Comm &worldComm,
                         const barney_api::mpi::Comm &workerComm,
                         const std::vector<LocalSlot> &localSlots)
    : Context(localSlots,makeTopo(worldComm,workerComm,localSlots)),
      // dataGroupIDs,gpuIDs,
      //         isActiveWorker?workersComm.rank:0,
      //         isActiveWorker?workersComm.size:1),
      world(worldComm),
      workers(workerComm) //worldComm.split(!isPassiveNode(localSlots)))
  {
    bool dbg = FromEnv::get()->logConfig;
    if (FromEnv::enabled("two-stage")) {
      std::cout << "ENABLING TwoStage!" << std::endl;
      globalTraceImpl = new TwoStage(this);
    } else if (FromEnv::enabled("all2all")) {
      std::cout << "ENABLING ALL2ALL!" << std::endl;
      globalTraceImpl = new MPIAll2all(this);
    } else {
      globalTraceImpl = new RQSMPI(this);
    }
  }

  /*! create a frame buffer object suitable to this context */
  std::shared_ptr<barney_api::FrameBuffer>
  MPIContext::createFrameBuffer()
  {
    return std::make_shared<DistFB>(this,devices);
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
    auto _context = this;
    for (auto device : *_context->devices) {
      SetActiveGPU forDuration(device);
      {
      auto rc = cudaGetLastError();
      if (rc) {
        PING; PRINT(rc);
        PRINT(cudaGetErrorString(rc));
      }
      assert(rc == 0);
      }
    }
    DistFB *fb = (DistFB *)_fb;
    if (isActiveWorker) {
    for (auto device : *_context->devices) {
      SetActiveGPU forDuration(device);
      {
      auto rc = cudaGetLastError();
      if (rc) {
        PING; PRINT(rc);
        PRINT(cudaGetErrorString(rc));
      }
      assert(rc == 0);
      }
    }
      renderTiles(renderer,model,camera,fb);
    for (auto device : *_context->devices) {
      SetActiveGPU forDuration(device);
      {
      auto rc = cudaGetLastError();
      if (rc) {
        PING; PRINT(rc);
        PRINT(cudaGetErrorString(rc));
      }
      assert(rc == 0);
      }
    }
      finalizeTiles(fb);
    for (auto device : *_context->devices) {
      SetActiveGPU forDuration(device);
      {
      auto rc = cudaGetLastError();
      if (rc) {
        PING; PRINT(rc);
        PRINT(cudaGetErrorString(rc));
      }
      assert(rc == 0);
      }
    }
    }
    // ------------------------------------------------------------------
    // done rendering, let the frame buffer know about it, so it can
    // do whatever needs doing with the latest finalized tiles
    // ------------------------------------------------------------------
    fb->finalizeFrame();
    for (auto device : *_context->devices) {
      SetActiveGPU forDuration(device);
      {
      auto rc = cudaGetLastError();
      if (rc) {
        PING; PRINT(rc);
        PRINT(cudaGetErrorString(rc));
      }
      assert(rc == 0);
      }
    }
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
                           // barney_api::mpi::Comm workers,
                           // bool isActiveWorker,
                           const std::vector<int> &dgIDs,
                           int numGPUs, const int *gpuIDs)
    {
      if (FromEnv::get()->logBackend)
        std::cout << "#bn: creating *optix* context" << std::endl;
      // std::vector<int> gpuIDs;
      int numDGs = dgIDs.size();
      if (numGPUs == -1) {
        BARNEY_CUDA_CALL(GetDeviceCount(&numGPUs));
        // int n = std::max(1,numGPUs/int(dgIDs.size()));
        // for (int i=0;i<n*dgIDs.size();i++)
        //   gpuIDs.push_back(i%numGPUs);
        // numGPUs = n*numGDdgIDs.size();
      }
      // else
      //   for (int i=0;i<numGPUs;i++)
      //     gpuIDs.push_back(_gpuIDs?_gpuIDs[i]:(i%numGPUs));
      if (numGPUs < numDGs)
        throw std::runtime_error
          ("not enough CUDA GPUs for requested number of data groups!");
      int gpusPerDG = numGPUs / numDGs;
      std::vector<LocalSlot> localSlots(dgIDs.size());
      for (int lsIdx=0;lsIdx<dgIDs.size();lsIdx++) {
        LocalSlot &slot = localSlots[lsIdx];
        slot.dataRank = dgIDs[lsIdx];
        for (int j=0;j<gpusPerDG;j++) {
          int idx = lsIdx*gpusPerDG+j;
          slot.gpuIDs.push_back(gpuIDs?gpuIDs[idx]:idx);
        }
      }

      barney_api::mpi::Comm workers
        = world.split(!isPassiveNode(localSlots));
      return new BARNEY_NS::MPIContext(world,workers,localSlots);
      //                                  dgIDs,gpuIDs);
    }
# endif
  }
}
