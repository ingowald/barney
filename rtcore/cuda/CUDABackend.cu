#include "rtcore/cuda/CUDABackend.h"

namespace barney {
  namespace cuda {

    BaseBackend::BaseBackend()
    {
      cudaFree(0);
      BARNEY_CUDA_CALL(GetDeviceCount(&numPhysicalDevices));
    }
    
    BaseDevGroup::BaseDevGroup(BaseBackend *backend,
                               const std::vector<int> &gpuIDs,
                               size_t sizeOfGlobals)
      : rtc::DevGroup(backend)
    {}
    

    void BaseBackend::setActiveGPU(int physicalID) 
    {
      BARNEY_CUDA_CHECK(cudaSetDevice(physicalID));
    }
    
    int  BaseBackend::getActiveGPU()
    {
      int current = 0;
      BARNEY_CUDA_CHECK(cudaGetDevice(&current));
      return current;
    }


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

  
  



