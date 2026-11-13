// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/DeviceGroup.h"
#include "native/render/Sampler.h"

namespace BARNEY_NS {
  namespace native {
    
    struct SamplerRegistry {
      typedef std::shared_ptr<SamplerRegistry> SP;
    
      SamplerRegistry(Device *device);
      ~SamplerRegistry();
      
      int allocate();
      void release(int nowReusableID);
      void grow();
    
      void setDD(int samplerID, const Sampler::DD &);

      Sampler::DD *getDD() 
      { return (Sampler::DD *)rtc.memory; }

      int numReserved = 0;
      int nextFree = 0;
      std::stack<int> reusableIDs;
      
      // struct PLD {
      //   Sampler::DD *memory = 0;
      // };
      // PLD *getPLD(Device *);
      // std::vector<PLD> perLogical;

      struct {
        Sampler::DD *memory = 0;
      } rtc;
      Device *const device;
      // DevGroup::SP const devices;
    };

  }
}
