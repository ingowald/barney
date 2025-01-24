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

#include "barney/Context.h"
#include "barney/render/SamplerRegistry.h"

namespace barney {
  namespace render {

    SamplerRegistry::SamplerRegistry(const DevGroup::SP &devices)
      : devices(devices)
    {
      numReserved = 1;
      perLogical.resize(devices->numLogical);
      
      for (auto device : *devices)
        getPLD(device)->buffer
          = device->rtc->createBuffer(numReserved*sizeof(Sampler::DD));
    }

    SamplerRegistry::~SamplerRegistry()
    {
      for (auto device : *devices)
        device->rtc->freeBuffer(getPLD(device)->buffer);
    }
     
    SamplerRegistry::PLD *SamplerRegistry::getPLD(Device *device)
    {
      return &perLogical[device->contextRank];
    }
    
    void SamplerRegistry::grow()
    {
      size_t oldNumBytes = numReserved * sizeof(Sampler::DD);
      numReserved *= 2;
      size_t newNumBytes = numReserved * sizeof(Sampler::DD);
      
      for (auto device : *devices) {
        // ------------------------------------------------------------------
        // save old samplers
        // ------------------------------------------------------------------
        PLD *pld = getPLD(device);
        rtc::Buffer *oldBuffer = pld->buffer;
        auto rtc = device->rtc;
        
        rtc::Buffer *newBuffer
          = rtc->createBuffer(newNumBytes);
        rtc->copyAsync(newBuffer->getDD(),oldBuffer->getDD(),oldNumBytes);
        rtc->sync();
        rtc->freeBuffer(oldBuffer);
        pld->buffer = newBuffer;
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
      size_t offset = samplerID*sizeof(dd);
      getPLD(device)->buffer->upload(&dd,sizeof(dd),offset);
    }

    
    
  }
}

