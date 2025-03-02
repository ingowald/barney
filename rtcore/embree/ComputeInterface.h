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
    inline __rtc_device T tex1D(rtc::device::TextureObject to,
                            float x);
    
    template<>
    inline __rtc_device float tex1D<float>(rtc::device::TextureObject to,
                                       float x)
    {
      return ((TextureSampler *)to)->tex1D(x).x;
    }
    template<>
    inline __rtc_device vec4f tex1D<vec4f>(rtc::device::TextureObject to,
                                       float x)
    {
      return ((TextureSampler *)to)->tex1D(x);
    }



    
    template<typename T>
    inline __rtc_device T tex2D(rtc::device::TextureObject to,
                            float x, float y);

    template<>
    inline __rtc_device float tex2D<float>(rtc::device::TextureObject to,
                                       float x, float y)
    {
      return ((TextureSampler *)to)->tex2D({x,y}).x;
    }

    template<>
    inline __rtc_device vec4f tex2D<vec4f>(rtc::device::TextureObject to,
                                         float x, float y)
    {
      return ((TextureSampler *)to)->tex2D({x,y});
    }



    
    template<typename T>
    inline __rtc_device T tex3D(rtc::device::TextureObject to,
                                       float x, float y, float z);
    
    template<>
    inline __rtc_device
    float tex3D<float>(rtc::device::TextureObject to,
                       float x, float y, float z)
    {
      return ((TextureSampler *)to)->tex3D({x,y,z}).x;
    }


    template<>
    inline __rtc_device
    vec4f tex3D<vec4f>(rtc::device::TextureObject to,
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
    
    
    // struct Compute : public rtc::Compute
    // {
    //   typedef void (*ComputeFct)(ComputeInterface &,
    //                              const void *dd);

    //   Compute(Device *device, const std::string &name);

    //   void launch(int numBlocks,
    //               int blockSize,
    //               const void *dd) override;
      
    //   void launch(vec2i numBlocks,
    //               vec2i blockSize,
    //               const void *dd) override;
      
    //   void launch(vec3i numBlocks,
    //               vec3i blockSize,
    //                       const void *dd) override;

    //   std::string const name;
    //   ComputeFct computeFct = 0;
    // };
    
    // struct Trace : public rtc::Trace
    // {
    //   typedef void (*TraceFct)(TraceInterface &);
    //   Trace(Device *device,
    //         const std::string &name);
      
    //   void launch(vec2i launchDims,
    //               const void *dd) override;
    //   void launch(int launchDims,
    //               const void *dd) override;
      
    //   void sync() override
    //   { /* no-op */ }
    //   TraceFct traceFct = 0;
    // };



    inline int atomicCAS(int *ptr, int _expected, int newValue)
    {
      int expected = _expected;
      ((std::atomic<int>*)ptr)->compare_exchange_weak(expected,newValue);
      return expected;
    }
  
    inline
    void fatomicMin(float *addr, float value)
    {
      float old = *(volatile float *)addr;
      if(old <= value) return;

      int _expected = (const int &)old;
      int _desired  = (const int &)value;
      while (true) {
        uint32_t _found = atomicCAS((int*)addr,_expected,_desired);
        if (_found == _expected)
          // write went though; we _did_ write the new mininm and
          // are done.
          return;
        // '_expected' changed, so write did not go through, and
        // somebody else wrote something new to that location.
        old = (const float &)_found;
        if (old <= value)
          // somebody else wrote something that's already smaller
          // than what we have ... leave it be, and done.
          return;
        else {
          // somebody else wrote something, but ours is _still_ smaller.
          _expected = _found;
          continue;
        }
      } 
    }

    inline
    void fatomicMax(float *addr, float value)
    {
      float old = *(volatile float *)addr;
      if(old >= value) return;

      int _expected = (const int &)old;
      int _desired  = (const int &)value;
      while (true) {
        uint32_t _found = atomicCAS((int*)addr,_expected,_desired);
        if (_found == _expected)
          // write went though; we _did_ write the new mininm and
          // are done.
          return;
        // '_expected' changed, so write did not go through, and
        // somebody else wrote something new to that location.
        old = (const float &)_found;
        if (old >= value)
          // somebody else wrote something that's already smaller
          // than what we have ... leave it be, and done.
          return;
        else {
          // somebody else wrote something, but ours is _still_ smaller.
          _expected = _found;
          continue;
        }
      } 
    }


    
  }
}
