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

#include "rtcore/embree/ComputeInterface.h"
#include "rtcore/embree/ComputeKernel.h"
#include <owl/common/parallel/parallel_for.h>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <barrier>
#include "rtcore/embree/TraceInterface.h"

namespace rtc {
  namespace embree {

    struct Task {
      virtual void run(int) = 0;
    };
    template<typename T>
    struct TaskWrapper : public Task {
      TaskWrapper(const T &t) : t(t) {}
      void run(int tid) override { t(tid); }
      const T t;
    };
    
    struct LaunchSystem {
      LaunchSystem();
      void launchAndWait(int numTotal, Task *task);
      void threadFct();

      std::vector<std::thread> threads;

      struct {
        volatile int total = 0;
        std::atomic<int> taken;
      } numJobs;
      
      Task *volatile task = 0;

      std::mutex mutex;
      std::barrier<void(*)() noexcept> barrier;
    };

    int numThreads() {
      int nt = std::thread::hardware_concurrency();
#if 0
      int maxNumThreads = 1;
#else
      int maxNumThreads = 1<<30;
#endif
      return std::min(nt,maxNumThreads);
    }
    
    LaunchSystem::LaunchSystem()
      : barrier(numThreads()+1,[]() noexcept {})
    {
      threads.reserve(numThreads());
      for (int i=0;i<numThreads();i++)
        threads.emplace_back([this](){ this->threadFct(); });

      barrier.arrive_and_wait();
    }
    
    void LaunchSystem::launchAndWait(int numTotal, Task *task)
    {
      // enqueue job
      std::lock_guard<std::mutex> lock(mutex);
      
      // workers are waiting on barrier here
      this->task = task;
      this->numJobs.total = numTotal;
      this->numJobs.taken = 0;
      
      this->barrier.arrive_and_wait();
      // workers do the work here

      this->barrier.arrive_and_wait();
    }

    LaunchSystem *createLaunchSystem() { return new LaunchSystem; }
    
    void LaunchSystem::threadFct()
    {
      barrier.arrive_and_wait();
      while (true) {

        // ------------------------------------------------------------------
        // wait for control thread to submit work
        // ------------------------------------------------------------------
        barrier.arrive_and_wait();
        
        // ------------------------------------------------------------------
        // run the actual task
        // ------------------------------------------------------------------
        while (true) {
          int tid = numJobs.taken++;
          if (tid >= numJobs.total)
            break;
          task->run(tid);
        }
        
        // ------------------------------------------------------------------
        // wait for all to be done
        // ------------------------------------------------------------------
        barrier.arrive_and_wait();
      }
    }
    
    template<typename Kernel>
    void parallel_for_3D(Device *device, vec3ui dims, const Kernel &lambda)
    {
      LaunchSystem *ls = ((embree::Device *)device)->ls;
      int numTotal = dims.x*dims.y*dims.z;
      TaskWrapper task([&](int tid)
      {
        // printf("in parallel %i / %i %i %i\n",tid,dims.x,dims.y,dims.z);
        int ix = tid % dims.x;
        int iz = tid / (dims.x * dims.y);
        int iy = (tid / dims.x) % dims.y;
        lambda(vec3ui(ix,iy,iz));
      });
      ls->launchAndWait(numTotal,&task);
    }
    
                         
    
    void ComputeKernel1D::launch(unsigned int numBlocks,
                                 unsigned int blockSize,
                                 const void *dd)
    {
      parallel_for_3D
        (device,vec3ui(numBlocks,1,1),
         // (numBlocks,
         [&](vec3ui bid){
           embree::ComputeInterface ci;
           ci.gridDim = vec3ui(numBlocks,1u,1u);
           ci.blockIdx = bid;
           ci.blockDim = vec3ui(blockSize,1u,1u);
           ci.threadIdx = vec3ui(0);
           for (ci.threadIdx.x=0;ci.threadIdx.x<blockSize;ci.threadIdx.x++)
             computeFct(ci,dd);
         });
    }
    
    void ComputeKernel2D::launch(vec2ui numBlocks,
                                 vec2ui blockSize,
                                 const void *dd)
    {
      parallel_for_3D
        (device,vec3ui(numBlocks.x,numBlocks.y,1),
         // (numBlocks,
         [&](vec3ui bid){
           embree::ComputeInterface ci;
           ci.gridDim = vec3ui(numBlocks.x,numBlocks.y,1);
           ci.blockIdx = bid;
           ci.blockDim = vec3ui(blockSize.x,blockSize.y,1);
           ci.threadIdx = vec3ui(0);
           for (ci.threadIdx.y=0;ci.threadIdx.y<blockSize.y;ci.threadIdx.y++)
             for (ci.threadIdx.x=0;ci.threadIdx.x<blockSize.x;ci.threadIdx.x++)
               computeFct(ci,dd);
         });
    }

    void ComputeKernel3D::launch(vec3ui numBlocks,
                                 vec3ui blockSize,
                                 const void *dd)
    {
      parallel_for_3D
        (device,numBlocks,
         // (numBlocks,
         [&](vec3ui bid){
           embree::ComputeInterface ci;
           ci.gridDim = vec3ui(numBlocks);
           ci.blockIdx = bid;
           ci.blockDim = vec3ui(blockSize);
           ci.threadIdx = vec3ui(0);
           for (ci.threadIdx.z=0;ci.threadIdx.z<blockSize.z;ci.threadIdx.z++)
             for (ci.threadIdx.y=0;ci.threadIdx.y<blockSize.y;ci.threadIdx.y++)
               for (ci.threadIdx.x=0;ci.threadIdx.x<blockSize.x;ci.threadIdx.x++)
                 computeFct(ci,dd);
         });
    }
      

    void TraceKernel2D::launch(vec2i launchDims,
                               const void *dd) 
    {
      parallel_for_3D
        (device,vec3ui(launchDims.x,launchDims.y,1),
         [&](vec3ui bid) {
           int ix = bid.x;
           int iy = bid.y;
           embree::TraceInterface ci;
           ci.launchIndex = vec3i(ix,iy,0);
           ci.launchDimensions = {launchDims.x,launchDims.y,1};
           ci.lpData = dd;
           kernelFct(ci);
         });
    }
      
  }
}
