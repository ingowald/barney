// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/Context.h"
#include "native/material/DeviceMaterial.h"
#include "native/material/HostMaterial.h"
#include "native/render/MaterialRegistry.h"

namespace BARNEY_NS {
  namespace native {

    // MaterialRegistry::PLD *MaterialRegistry::getPLD(Device *device)
    // { return &perLogical[device->contextRank()]; }
    
    MaterialRegistry::MaterialRegistry(Device *device)
      : device(device)
    {
      numReserved = 1;
      rtc.buffer
        = device->rtc->createBuffer(numReserved*sizeof(DeviceMaterial));
    }

    MaterialRegistry::~MaterialRegistry()
    {
      // owlBufferRelease(buffer);
      device->rtc->freeBuffer(rtc.buffer);
    }
    
    void MaterialRegistry::grow()
    {
      size_t oldNumBytes = numReserved * sizeof(DeviceMaterial);
      numReserved *= 2;
      size_t newNumBytes = numReserved * sizeof(DeviceMaterial);
      // ------------------------------------------------------------------
      // save old materials
      // ------------------------------------------------------------------
      auto pld = &this->rtc;
      rtc::Buffer *oldBuffer = pld->buffer;
      auto rtc = device->rtc;
      
      rtc::Buffer *newBuffer
        = rtc->createBuffer(newNumBytes);
      rtc->copyAsync(newBuffer->getDD(),oldBuffer->getDD(),oldNumBytes);
      rtc->sync();
      rtc->freeBuffer(oldBuffer);
      pld->buffer = newBuffer;
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
                                       const DeviceMaterial &dd)
    {
      // for (auto device : *devices) {
      rtc.buffer->upload(&dd,sizeof(dd),sizeof(dd)*materialID);
      // }
    }


    
  }
}
