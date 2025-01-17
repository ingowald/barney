#pragma once

#include "rtcore/common/Backend.h"

namespace barney {
  namespace embree {
    struct EmbreeBackend;
    
    struct DevGroup : public rtc::DevGroup {
      DevGroup(EmbreeBackend *backend,
               const std::vector<int> &gpuIDs,
               size_t sizeOfGlobals);
    };

    struct EmbreeBackend : public rtc::Backend {
      EmbreeBackend();
      
      void setActiveGPU(int physicalID) override;
      int  getActiveGPU() override;
      rtc::DevGroup *createDevGroup(const std::vector<int> &gpuIDs,
                                    size_t sizeOfGlobals) override;
    };
    
  }
}

  
  
