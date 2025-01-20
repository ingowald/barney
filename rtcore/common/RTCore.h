#pragma once

#include "barney/common/barney-common.h"

namespace barney {
  namespace rtc {

    typedef struct _OpaqueTextureHandle *OpaqueTextureHandle;
  }
  namespace cuda {

#ifdef __CUDACC__
    struct ComputeInterface
    {
      inline __device__ vec3ui getThreadIdx() const
      { return threadIdx; }
      inline __device__ vec3ui getBlockDim() const
      { return {blockDim.x,blockDim.y,blockDim.z}; }
      inline __device__ vec3ui getBlockIdx() const
      { return {blockIdx.x,blockIdx.y,blockIdx.z}; }
      inline __device__ int atomicAdd(int *ptr, int inc) const
      { return ::atomicAdd(ptr,inc); }
      inline __device__ float atomicAdd(float *ptr, float inc) const
      { return ::atomicAdd(ptr,inc); }
    };
    
    template<typename KernelT>
    __global__ void
    runKernel(KernelT dd)
    {
      dd.run(::barney::cuda::ComputeInterface());
    }
    
#define RTC_CUDA_COMPUTE_KERNEL(KernelName,ClassName)                   \
    extern "C" void                                                     \
    barney_rtc_cuda_launch_##KernelName(::barney::vec3ui nb,            \
                                        ::barney::vec3ui bs,            \
                                        int shmSize,                    \
                                        cudaStream_t stream,            \
                                        const void *dd)                 \
    {                                                                   \
      barney::cuda::runKernel                                           \
        <<<dim3(nb.x,nb.y,nb.z),dim3(bs.x,bs.y,bs.z),shmSize,stream>>>  \
        (*(typename ClassName *)dd);                                    \
    }
  } // ::barney::cuda

  namespace optix {
    struct RTCoreInterface : public cuda::ComputeInterface {
      inline __device__ vec3i getLaunchDims()  const
      { return optixGetLaunchDimensions(); }
      inline __device__ vec3i getLaunchIndex() const
      { return optixGetLaunchIndex(); }
      template<typename PRDType>
      inline __device__ void traceRay(rtc::device::AccelHandle world,
                                       vec3f org,
                                       vec3f dir,
                                       float t0,
                                       float t1,
                                       PRDType &prd) const
      {
        owl::traceRay((const OptixTraversableHandle &)world,
                             owl::Ray(org,dir,t0,t1),prd);
      }
    };
    
#define RTC_OPTIX_TRACE_KERNEL(KernelName,RayGenType,LaunchParamsType)  \
    extern "C" __global__ void                                          \
    __raygen__##KernelName()                                            \
    {                                                                   \
      ::barney::optix::RTCoreInterface rtcore;                          \
      RayGenType *rg = (RayGenType*)optixGetSbtDataPointer();           \
      rg->run(rtcore);                                                  \
    }
#endif
  } // ::barney::optix
}

namespace barney {
  namespace rtc {
    // TODO:
    template<typename T>
    inline __device__ __host__ T tex1D(barney::rtc::device::TextureObject to,
                                       float x)
    {
#ifdef __CUDA_ARCH__
      printf("tex2d missing...\n");
      return T{};
#else
      BARNEY_NYI();
#endif
    }
    template<typename T>
    inline __both__ T tex2D(barney::rtc::device::TextureObject to,
                                 float x, float y);


    template<>
    inline __both__ float4 tex2D<float4>(barney::rtc::device::TextureObject to,
                                         float x, float y)
    {
#ifdef __CUDA_ARCH__
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      float4 v = ::tex2D<float4>(texObj,x,y);
      // printf("%p %f %f -> %f %f %f %f\n",
      //        (int*)texObj,x,y,v.x,v.y,v.z,v.w);
      return v;
      // return T{};
#else
      BARNEY_NYI();
#endif
    }

    template<>
    inline __both__ float tex2D<float>(barney::rtc::device::TextureObject to,
                                       float x, float y)
    {
#ifdef __CUDA_ARCH__
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      return ::tex2D<float>(texObj,x,y);
      // return T{};
#else
      BARNEY_NYI();
#endif
    }


    template<typename T>
    inline __device__ __host__ T tex3D(barney::rtc::device::TextureObject to,
                                       float x, float y, float z)
    {
#ifdef __CUDA_ARCH__
      printf("tex2d missing...\n");
      return T{};
#else
      BARNEY_NYI();
#endif
    }
  }
}
