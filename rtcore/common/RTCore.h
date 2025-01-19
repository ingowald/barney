#pragma once

#include "barney/common/barney-common.h"

namespace barney {
  namespace rtc {

    typedef struct _OpaqueTextureHandle *OpaqueTextureHandle;
  }
  namespace cuda {

#ifdef __CUDACC__
    struct ComputeKernelInterface
    {
      inline __device__ vec3ui getThreadIdx() const
      { return threadIdx; }
      inline __device__ vec3ui getBlockDim() const
      { return {blockDim.x,blockDim.y,blockDim.z}; }
      inline __device__ vec3ui getBlockIdx() const
      { return {blockIdx.x,blockIdx.y,blockIdx.z}; }
      inline __device__ int atomicAdd(int *ptr, int inc) const
      { return ::atomicAdd(ptr,inc); }
    };
    
    template<typename KernelT>
    __global__ void
    runKernel(KernelT dd)
    {
      dd.run(::barney::cuda::ComputeKernelInterface());
    }
    
#define RTC_CUDA_DEFINE_COMPUTE(KernelName,ClassName)                   \
    extern "C" void                                                     \
    barney_rtc_cuda_launch_##KernelName(::barney::vec3ui nb,            \
                                        ::barney::vec3ui bs,              \
                                        int shmSize,                    \
                                        cudaStream_t stream,            \
                                        const void *dd)                 \
    {                                                                   \
      barney::cuda::runKernel                                           \
        <<<dim3(nb.x,nb.y,nb.z),dim3(bs.x,bs.y,bs.z),shmSize,stream>>>  \
        (*(typename ClassName *)dd);                                \
    }
#endif
  }
}
