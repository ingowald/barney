#pragma once

#include "barney/common/barney-common.h"
#ifdef __CUDA_ARCH__
#include "owl/owl_device.h"
#endif

namespace barney {
  namespace rtc {

    typedef struct _OpaqueTextureHandle *OpaqueTextureHandle;
  }
#ifdef __CUDACC__
  namespace cuda {

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
    
#define RTC_CUDA_COMPUTE(KernelName,ClassName)                          \
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
#endif


  
#if BARNEY_COMPILE_OPTIX_PROGRAMS
  namespace optix {
    inline __device__ const void *getLaunchParamsPointer();
    
    struct RTCoreInterface : public cuda::ComputeInterface {
      // inline __device__ RTCoreInterface(const void *globalsMem)
      //   : globalsMem(globalsMem)
      // {}
      inline __device__ void ignoreIntersection() const
      { optixIgnoreIntersection(); }
      
      inline __device__ void reportIntersection(float t, int i) const
      { optixReportIntersection(t,i); }

      inline __device__ void *getPRD() const
      { return ::owl::getPRDPointer(); }
      
      inline __device__ const void *getProgramData() const
      { return (const void *)optixGetSbtDataPointer(); }
      
      inline __device__ const void *getLPData() const
      { return getLaunchParamsPointer(); }
      
      inline __device__ vec3i getLaunchDims()  const
      { return optixGetLaunchDimensions(); }
      
      inline __device__ vec3i getLaunchIndex() const
      { return optixGetLaunchIndex(); }

      inline __device__ vec2f getTriangleBarycentrics() const
      { return optixGetTriangleBarycentrics(); }
      
      inline __device__ int getPrimitiveIndex() const
      { return optixGetPrimitiveIndex(); }
      
      inline __device__ float getRayTmax() const
      { return optixGetRayTmax(); }
      
      inline __device__ float getRayTmin() const
      { return optixGetRayTmin(); }

      inline __device__ vec3f getObjectRayDirection() const
      { return optixGetObjectRayDirection(); }
      
      inline __device__ vec3f getObjectRayOrigin() const
      { return optixGetObjectRayOrigin(); }
      
      inline __device__ vec3f getWorldRayDirection() const
      { return optixGetWorldRayDirection(); }
      
      inline __device__ vec3f getWorldRayOrigin() const
      { return optixGetWorldRayOrigin(); }
      
      inline __device__
      vec3f transformNormalFromObjectToWorldSpace(vec3f v) const
      { return optixTransformNormalFromObjectToWorldSpace(v); }
      
      inline __device__
      vec3f transformPointFromObjectToWorldSpace(vec3f v) const
      { return optixTransformPointFromObjectToWorldSpace(v); }
      
      inline __device__
      vec3f transformVectorFromObjectToWorldSpace(vec3f v) const
      { return optixTransformVectorFromObjectToWorldSpace(v); }
      
      inline __device__
      vec3f transformNormalFromWorldToObjectSpace(vec3f v) const
      { return optixTransformNormalFromWorldToObjectSpace(v); }
      
      inline __device__
      vec3f transformPointFromWorldToObjectSpace(vec3f v) const
      { return optixTransformPointFromWorldToObjectSpace(v); }
      
      inline __device__
      vec3f transformVectorFromWorldToObjectSpace(vec3f v) const
      { return optixTransformVectorFromWorldToObjectSpace(v); }
      
      
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
      // const void *const globalsMem;
    };

#define RTC_DECLARE_GLOBALS(Type)                               \
    __constant__ Type optixLaunchParams;                        \
    namespace barney {                                          \
      namespace optix {                                         \
        inline __device__ const void *getLaunchParamsPointer()  \
        { return &optixLaunchParams; }                          \
      }                                                         \
    }

# define RTC_OPTIX_TRACE_KERNEL(name,RayGenType)                \
    extern "C" __global__                                       \
    void __raygen__##name()                                     \
    {                                                           \
      RayGenType *rg = (RayGenType*)optixGetSbtDataPointer();   \
      ::barney::optix::RTCoreInterface rtcore;                  \
      rg->run(rtcore);                                          \
    }                                                                   
    

#define RTC_DECLARE_USER_GEOM(name,type)                        \
                                                                \
    extern "C" __global__                                       \
    void __closesthit__##name() {                               \
      ::barney::optix::RTCoreInterface rtcore;                  \
      type::closest_hit(rtcore);                                \
    }                                                           \
                                                                \
    extern "C" __global__                                       \
    void __anyhit__##name() {                                   \
      ::barney::optix::RTCoreInterface rtcore;                  \
      type::any_hit(rtcore);                                    \
    }                                                           \
                                                                \
    extern "C" __global__                                       \
    void __insersect__##name() {                                \
      ::barney::optix::RTCoreInterface rtcore;                  \
      type::intersect(rtcore);                                  \
    }                                                           \
                                                                \
    extern "C" __global__                                       \
    void __boundsFunc__##name(const void *geom,                 \
                              owl::common::box3f &result,       \
                              int primID)                       \
    {                                                           \
      ::barney::optix::RTCoreInterface rtcore;                  \
      type::bounds(rtcore,geom,result,primID);                  \
    }                                                           \
  
#define RTC_DECLARE_TRIANGLES_GEOM(name,type)   \
                                                \
    extern "C" __global__                       \
    void __closesthit__##name() {               \
      ::barney::optix::RTCoreInterface rtcore;  \
      type::closest_hit(rtcore);                \
    }                                           \
                                                \
    extern "C" __global__                       \
    void __anyhit__##name() {                   \
      ::barney::optix::RTCoreInterface rtcore;  \
      type::any_hit(rtcore);                    \
    }                                           \
  
  } // ::barney::optix
#else
# define RTC_DECLARE_GLOBALS(Type) /* nothing */
# define RTC_OPTIX_TRACE_KERNEL(name,RayGenType) /* nothing */
#endif  
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
