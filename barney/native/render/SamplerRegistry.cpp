// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/Context.h"
#include "native/render/SamplerRegistry.h"

namespace BARNEY_NS {
  namespace native {

    SamplerRegistry::SamplerRegistry(Device *device)
      : device(device)
    {
      numReserved = 1;

      SetActiveGPU forDuration(device);
      rtc.memory
        = (Sampler::DD*)device->rtc->allocMem(numReserved*sizeof(Sampler::DD));
      
      if (size_t(rtc.memory) % 16)
        throw std::runtime_error("sampler mem not aligned...");
    }

    SamplerRegistry::~SamplerRegistry()
    {
      SetActiveGPU forDuration(device);
      device->rtc->freeMem(rtc.memory);
    }
     
    SamplerRegistry::PLD *SamplerRegistry::getPLD(Device *device)
    {
      return &perLogical[device->contextRank()];
    }
    
    void SamplerRegistry::grow()
    {
      size_t oldNumBytes = numReserved * sizeof(Sampler::DD);
      numReserved *= 2;
      size_t newNumBytes = numReserved * sizeof(Sampler::DD)+128;
      
      SetActiveGPU forDuration(device);
      // ------------------------------------------------------------------
      // save old samplers
      // ------------------------------------------------------------------
      auto pld = &this->rtc;
      auto rtc = device->rtc;
      
      Sampler::DD *oldMem = pld->memory;
      pld->memory = (Sampler::DD*)rtc->allocMem(newNumBytes);
      rtc->copy(pld->memory,oldMem,oldNumBytes);
      rtc->freeMem(oldMem);
    }

    int SamplerRegistry::allocate()
    {
      if (!reusableIDs.empty()) {
        int ID = reusableIDs.top();
        reusableIDs.pop();
        return ID;
      }
      if (nextFree == numReserved) grow();

      return nextFree++;
    }
   
    void SamplerRegistry::release(int nowReusableID)
    {
      reusableIDs.push(nowReusableID);
    }
  
    void SamplerRegistry::setDD(int samplerID,
                                const Sampler::DD &dd)
    {
      SetActiveGPU forDuration(device);
      size_t offset = samplerID*sizeof(dd);
      assert(samplerID >= 0 && samplerID < nextFree);
      device->rtc->copy(((uint8_t*)getPLD(device)->memory)+offset,
                        &dd,sizeof(dd));
    }
    
  }
}

