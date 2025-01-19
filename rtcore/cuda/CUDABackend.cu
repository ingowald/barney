#include "rtcore/cuda/CUDABackend.h"

namespace barney {
  namespace cuda {

    rtc::DevGroup *CUDABackend
    ::createDevGroup(const std::vector<int> &gpuIDs,
                     size_t sizeOfGlobals)
    {
      BARNEY_NYI();
    }
    
  }

  namespace rtc {
    Backend *createBackend_cuda()
    {
      return new barney::cuda::CUDABackend;
    }
  }
}

  
  



