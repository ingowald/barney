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
#include "rtcore/common/RTCore.h"
#include <owl/common/parallel/parallel_for.h>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <barrier>

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
          // printf("started %i/%i\n",tid,numJobs.total);
          task->run(tid);
          // printf("finished %i/%i\n",tid,numJobs.total);
        }
        
        // ------------------------------------------------------------------
        // wait for all to be done
        // ------------------------------------------------------------------
        barrier.arrive_and_wait();
      }
    }
    
    template<typename Kernel>
    void parallel_for_3D(Device *device, vec3i dims, const Kernel &lambda)
    {
      LaunchSystem *ls = ((embree::Device *)device)->ls;
      int numTotal = dims.x*dims.y*dims.z;
      TaskWrapper task([&](int tid)
      {
        // printf("in parallel %i / %i %i %i\n",tid,dims.x,dims.y,dims.z);
        int ix = tid % dims.x;
        int iz = tid / (dims.x * dims.y);
        int iy = (tid / dims.x) % dims.y;
        lambda(vec3i(ix,iy,iz));
      });
      ls->launchAndWait(numTotal,&task);
    }
    
                         
    
    // Compute::Compute(Device *device,
    //                  const std::string &name)
    //   : rtc::Compute(device),
    //     name(name)
    // {
    //   computeFct = (ComputeFct)rtc::getSymbol
    //     ("barney_rtc_embree_computeBlock_"+name);
    // }

    // void Compute::launch(int numBlocks,
    //                      int blockSize,
    //                      const void *dd)
    // {
    //   launch(vec3i(numBlocks,1,1),
    //          vec3i(blockSize,1,1),
    //          dd);
    // }

    // void Compute::launch(vec2i numBlocks,
    //                      vec2i blockSize,
    //                      const void *dd)
    // {
    //   launch(vec3i(numBlocks.x,numBlocks.y,1),
    //          vec3i(blockSize.x,blockSize.y,1),
    //          dd);
    // }

    // void Compute::launch(vec3i numBlocks,
    //                      vec3i blockSize,
    //                      const void *dd)
    // {
    //   parallel_for_3D
    //     (device,numBlocks,
    //      // (numBlocks,
    //      [&](vec3i bid){
    //        embree::ComputeInterface ci;
    //        ci.gridDim = vec3ui(numBlocks);//vec3ui(numBlocks,1,1);
    //        ci.blockIdx = vec3ui(bid);//vec3ui(b,0,0);
    //        ci.blockDim = vec3ui(blockSize);//vec3ui(blockSize,1,1);
    //        ci.threadIdx = vec3ui(0);
    //        computeFct(ci,dd);
    //      });
    // }
      

    // Trace::Trace(Device *device,
    //              const std::string &name)
    //   : rtc::Trace(device)
    // { 
    //   traceFct = (TraceFct)rtc::getSymbol
    //     ("barney_rtc_embree_trace_"+name);
    // }
    
    // void Trace::launch(vec2i launchDims,
    //                    const void *dd) 
    // {
    //   parallel_for_3D
    //     (device,vec3i(launchDims.x,launchDims.y,1),
    //      [&](vec3i bid) {
    //      // (numBlocks,
    //   // for (int iy=0;iy<launchDims.y;iy++)
    //   //   for (int ix=0;ix<launchDims.x;ix++)
    //        int ix = bid.x;
    //        int iy = bid.y;
    //        embree::TraceInterface ci;
    //        ci.launchIndex = vec3i(ix,iy,0);
    //        ci.launchDimensions = {launchDims.x,launchDims.y,1};
    //        ci.lpData = dd;
    //        traceFct(ci);
    //      });
    // }
      
    // void Trace::launch(int launchDims,
    //                    const void *dd) 
    // {
    //   launch(vec2i(launchDims,1),dd);
    // }

  }
}
