// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/DeviceGroup.h"
#include "native/render/Sampler.h"

namespace BARNEY_NS {
  namespace native {

    struct DeviceMaterial;
    
    struct MaterialRegistry {
      typedef std::shared_ptr<MaterialRegistry> SP;
    
      MaterialRegistry(Device *device);
      ~MaterialRegistry();
      
      int allocate();
      void release(int nowReusableID);
      void grow();

      void setMaterial(int materialID,
                       const DeviceMaterial &dd);
    
      int numReserved = 0;
      int nextFree = 0;
    
      std::stack<int> reusableIDs;
      // OWLBuffer       buffer = 0;

      DeviceMaterial *getDD() 
      { return (DeviceMaterial *)rtc.buffer->getDD(); }
      // { return (DeviceMaterial *)getPLD(device)->buffer->getDD(); }

      // struct PLD {
      //   rtc::Buffer *buffer = 0;
      //   // Device      *device;
      // };
      struct {
        rtc::Buffer *buffer = 0;
      } rtc;
      // PLD *getPLD(Device *device);
      // std::vector<PLD> perLogical;

      // DevGroup::SP const devices;
      Device *const device;
    };

  }
}
