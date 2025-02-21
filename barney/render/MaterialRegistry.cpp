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
#include "barney/material/DeviceMaterial.h"
#include "barney/material/Material.h"
#include "barney/render/MaterialRegistry.h"

namespace BARNEY_NS {
  namespace render {

    MaterialRegistry::PLD *MaterialRegistry::getPLD(Device *device)
    { return &perLogical[device->contextRank]; }
    
    MaterialRegistry::MaterialRegistry(const DevGroup::SP &devices)
      : devices(devices)
    {
      numReserved = 1;
      perLogical.resize(devices->numLogical);
      
      for (auto device : *devices)
        getPLD(device)->buffer
          = device->rtc->createBuffer(numReserved*sizeof(DeviceMaterial));
    }

    MaterialRegistry::~MaterialRegistry()
    {
      // owlBufferRelease(buffer);
      for (auto device : *devices)
        device->rtc->freeBuffer(getPLD(device)->buffer);
    }
    
    void MaterialRegistry::grow()
    {
      size_t oldNumBytes = numReserved * sizeof(DeviceMaterial);
      numReserved *= 2;
      size_t newNumBytes = numReserved * sizeof(DeviceMaterial);
      for (auto device : *devices) {
      // ------------------------------------------------------------------
      // save old materials
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

    int MaterialRegistry::allocate()
    {
      if (!reusableIDs.empty()) {
        int ID = reusableIDs.top();
        reusableIDs.pop();
        return ID;
      }
      if (nextFree == numReserved) grow();

      return nextFree++;
    }
   
    void MaterialRegistry::release(int nowReusableID)
    {
      reusableIDs.push(nowReusableID);
    }
  
    void MaterialRegistry::setMaterial(int materialID,
                                       const DeviceMaterial &dd,
                                       Device *device)
    {
      // for (auto device : *devices) {
        PLD *pld = getPLD(device);
        pld->buffer->upload(&dd,sizeof(dd),sizeof(dd)*materialID);
      // }
    }


    
  }
}
