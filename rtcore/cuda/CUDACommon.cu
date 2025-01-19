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
    

    int BaseDevice::setActive() const
    {
      int oldActive = 0;
      BARNEY_CUDA_CHECK(cudaGetDevice(&oldActive));
      BARNEY_CUDA_CHECK(cudaSetDevice(physicalID));
      return oldActive;
    }
    
    void BaseDevice::restoreActive(int oldActive) const
    {
      BARNEY_CUDA_CHECK(cudaSetDevice(oldActive));
    }
    
    
  }
}

