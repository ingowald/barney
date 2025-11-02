// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/Context.h"
#include "barney/render/SamplerRegistry.h"

namespace BARNEY_NS {
  namespace render {

    SamplerRegistry::SamplerRegistry(const DevGroup::SP &devices)
      : devices(devices)
    {
      numReserved = 1;
      perLogical.resize(devices->numLogical);
      
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        getPLD(device)->memory
          = (Sampler::DD*)device->rtc->allocMem(numReserved*sizeof(Sampler::DD));

        if (size_t(getPLD(device)->memory) % 16)
          throw std::runtime_error("sampler mem not aligned...");
      }
    }

    SamplerRegistry::~SamplerRegistry()
    {
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        device->rtc->freeMem(getPLD(device)->memory);
      }
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
      
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        // ------------------------------------------------------------------
        // save old samplers
        // ------------------------------------------------------------------
        PLD *pld = getPLD(device);
        auto rtc = device->rtc;
        
        Sampler::DD *oldMem = pld->memory;
        pld->memory = (Sampler::DD*)rtc->allocMem(newNumBytes);
        rtc->copy(pld->memory,oldMem,oldNumBytes);
        rtc->freeMem(oldMem);
      }
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
                                const Sampler::DD &dd,
                                Device *device)
    {
      SetActiveGPU forDuration(device);
      size_t offset = samplerID*sizeof(dd);
      assert(samplerID >= 0 && samplerID < nextFree);
      device->rtc->copy(((uint8_t*)getPLD(device)->memory)+offset,
                        &dd,sizeof(dd));
    }
    
  }
}

