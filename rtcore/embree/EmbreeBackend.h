#pragma once

#include "rtcore/common/Backend.h"

namespace barney {
  namespace embree {
    struct EmbreeBackend;
    
    struct EmbreeBackend : public rtc::Backend {
      EmbreeBackend();
      virtual ~EmbreeBackend();
      // void setActiveGPU(int physicalID) override;
      // int  getActiveGPU() override;
      rtc::Device *createDevice(int gpuID) override;
    };
    
  }
}

  
  
