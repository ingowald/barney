// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/DeviceGroup.h"
#include "barney/render/Sampler.h"

namespace BARNEY_NS {
  struct ModelSlot;
  
  namespace render {
    
    struct SamplerRegistry {
      typedef std::shared_ptr<SamplerRegistry> SP;
    
      SamplerRegistry(const DevGroup::SP &devices);
      virtual ~SamplerRegistry();
      
      int allocate();
      void release(int nowReusableID);
      void grow();
    
      void setDD(int samplerID, const Sampler::DD &, Device *device);

      Sampler::DD *getDD(Device *device) 
      { return (Sampler::DD *)getPLD(device)->memory; }

      int numReserved = 0;
      int nextFree = 0;
      std::stack<int> reusableIDs;
      
      struct PLD {
        Sampler::DD *memory = 0;
      };
      PLD *getPLD(Device *);
      std::vector<PLD> perLogical;
      
      DevGroup::SP const devices;
    };

  }
}
