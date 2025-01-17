#pragma once

#include "rtcore/common/Backend.h"
#include "barney/common/cuda-helper.h"

namespace barney {
  namespace cuda {

    struct BaseBackend;

    struct BaseDevice : public rtc::Device {
      /*! sets this gpu as active, and returns physical ID of GPU that
          was active before */
      int setActive() const override;
      
      /*! restores the gpu whose ID was previously returend by setActive() */
      void restoreActive(int oldActive) const override;
      
    };
      
    struct BaseDevGroup : public rtc::DevGroup {
      BaseDevGroup(BaseBackend *backend,
                   const std::vector<int> &gpuIDs,
                   size_t sizeOfGlobals);
    };
    
    struct BaseBackend : public rtc::Backend {
      BaseBackend();
      // void setActiveGPU(int physicalID) override;
      // int  getActiveGPU() override;
    };

    struct CUDABackend : public cuda::BaseBackend {
      rtc::DevGroup *createDevGroup(const std::vector<int> &gpuIDs,
                                    size_t sizeOfGlobals) override;
    };
    
  }
}

  
  
