// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/DeviceGroup.h"
// #include "barney/material/Globals.h"
// #include "barney/render/DeviceMaterial.h"
#include "barney/render/Sampler.h"

namespace BARNEY_NS {
  namespace render {

    struct DeviceMaterial;
    
    struct MaterialRegistry {
      typedef std::shared_ptr<MaterialRegistry> SP;
    
      MaterialRegistry(const DevGroup::SP &devices);
      virtual ~MaterialRegistry();
      
      int allocate();
      void release(int nowReusableID);
      void grow();

      void setMaterial(int materialID,
                       const DeviceMaterial &dd,
                       Device *device);
      // const DeviceMaterial *getPointer(int owlDeviceID) const;
    
      int numReserved = 0;
      int nextFree = 0;
    
      std::stack<int> reusableIDs;
      // OWLBuffer       buffer = 0;

      DeviceMaterial *getDD(Device *device) 
      { return (DeviceMaterial *)getPLD(device)->buffer->getDD(); }

      struct PLD {
        rtc::Buffer *buffer = 0;
        // Device      *device;
      };
      PLD *getPLD(Device *device);
      std::vector<PLD> perLogical;

      DevGroup::SP const devices;
    };

  }
}
