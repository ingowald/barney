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

#pragma once

#include "rtcore/embree/Device.h"
#include "rtcore/embree/Texture.h"
#include <atomic>

namespace rtc {
  namespace embree {

    struct ComputeInterface;
    struct TraceInterface;



    template<typename T>
    inline __rtc_device T tex1D(rtc::TextureObject to,
                                float x);
    
    template<>
    inline __rtc_device float tex1D<float>(rtc::TextureObject to,
                                           float x)
    {
      return ((TextureSampler *)to)->tex1D(x).x;
    }
    template<>
    inline __rtc_device vec4f tex1D<vec4f>(rtc::TextureObject to,
                                           float x)
    {
      return ((TextureSampler *)to)->tex1D(x);
    }



    
    template<typename T>
    inline __rtc_device T tex2D(rtc::TextureObject to,
                                float x, float y);

    template<>
    inline __rtc_device float tex2D<float>(rtc::TextureObject to,
                                           float x, float y)
    {
      return ((TextureSampler *)to)->tex2D({x,y}).x;
    }

    template<>
    inline __rtc_device vec4f tex2D<vec4f>(rtc::TextureObject to,
                                           float x, float y)
    {
      return ((TextureSampler *)to)->tex2D({x,y});
    }



    
    template<typename T>
    inline __rtc_device T tex3D(rtc::TextureObject to,
                                float x, float y, float z);
    
    template<>
    inline __rtc_device
    float tex3D<float>(rtc::TextureObject to,
                       float x, float y, float z)
    {
      return ((TextureSampler *)to)->tex3D({x,y,z}).x;
    }


    template<>
    inline __rtc_device
    vec4f tex3D<vec4f>(rtc::TextureObject to,
                       float x, float y, float z)
    {
      return ((TextureSampler *)to)->tex3D({x,y,z});
    }



    struct ComputeInterface {
      inline vec3ui launchIndex() const
      {
        return getThreadIdx() + getBlockIdx() * getBlockDim();
      }
      inline vec3ui getThreadIdx() const
      { return this->threadIdx; }
      
      inline vec3ui getBlockDim() const
      { return this->blockDim; }
      
      inline vec3ui getBlockIdx() const
      { return this->blockIdx; }
      
      inline int atomicAdd(int *ptr, int inc) const
      { return ((std::atomic<int> *)ptr)->fetch_add(inc); }
      
      inline float atomicAdd(float *ptr, float inc) const
      { return ((std::atomic<float> *)ptr)->fetch_add(inc); }
      
      vec3ui threadIdx;
      vec3ui blockIdx;
      vec3ui blockDim;
      vec3ui gridDim;
    };
    
    inline
    void fatomicMin(float *addr, float value)
    {
      float current = *(volatile float *)addr;
      while (current > value) {
        bool wasChanged
          = ((std::atomic<int>*)addr)->compare_exchange_weak((int&)current,(int&)value);
        if (wasChanged) break;
      }
    }

    inline
    void fatomicMax(float *addr, float value)
    {
      float current = *(volatile float *)addr;
      while (current < value) {
        bool wasChanged
          = ((std::atomic<int>*)addr)->compare_exchange_weak((int&)current,(int&)value);
        if (wasChanged) break;
      }
    }

    template<typename TASK_T>
    inline void serial_for(int nTasks, TASK_T&& taskFunction)
    {
      for (int taskIndex = 0; taskIndex < nTasks; ++taskIndex) {
        taskFunction(taskIndex);
      }
    }
    
  }
}

# define __rtc_global /*static*/
# define __rtc_launch(dev,kernel,nb,bs,...)                     \
  rtc::embree::serial_for(nb,[&](int taskID) {                  \
    rtc::embree::ComputeInterface ci;                           \
    ci.gridDim = {(unsigned)nb,1u,1u};                          \
    ci.blockDim = {(unsigned)bs,1u,1u};                         \
    ci.blockIdx = {(unsigned)taskID,0u,0u};                     \
    ci.threadIdx = {0u,0u,0u};                                  \
    for (ci.threadIdx.x=0;ci.threadIdx.x<bs;ci.threadIdx.x++){  \
      kernel(ci,__VA_ARGS__);                                   \
    }                                                           \
  });
