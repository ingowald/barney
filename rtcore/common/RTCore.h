#pragma once


#if 1
#else

#include "barney/barney.h"
#include "barney/common/barney-common.h"
#ifdef __CUDA_ARCH__
#include "owl/owl_device.h"
#endif
#include <atomic>

#if BARNEY_BACKEND_EMBREE
# include <embree4/rtcore_common.h>
# include <embree4/rtcore_ray.h>
#endif
 

#if defined(_MSC_VER)
//#  define BARNEY_VISIBILITY_DEFAULT /* nothing */
#  define BARNEY_VISIBILITY_DEFAULT __declspec(dllexport)
#elif defined(__clang__) || defined(__GNUC__)
# ifdef __CUDA_ARCH__
#  define BARNEY_VISIBILITY_DEFAULT /* nothing */
# else
#  define BARNEY_VISIBILITY_DEFAULT  __attribute__ ((visibility("default")))
# endif
#else
#  define BARNEY_VISIBILITY_DEFAULT /* nothing */
#endif


namespace barney {

  inline __both__ vec4f load(float4 v) { return vec4f(v.x,v.y,v.z,v.w); }
  


#if __CUDA_ARCH__
  using ::atomicCAS;
  using ::__int_as_float;
  using ::__float_as_int;
#else
  inline uint32_t
  __float_as_int(float f)
  {
    uint32_t ui;
    memcpy(&ui,&f,sizeof(f));
    return ui;
  }
  inline float __int_as_float(int i)
  {
    float f;
    memcpy(&f,&i,sizeof(f));
    return f;
  }
  inline int atomicCAS(int *ptr, int _expected, int newValue)
  {
    int expected = _expected;
    ((std::atomic<int>*)ptr)->compare_exchange_weak(expected,newValue);
    return expected;
  }
#endif 

#ifndef __CUDACC__
#endif
  
  
  inline __both__
  void fatomicMin(float *addr, float value)
  {
    float old = *(volatile float *)addr;
    if(old <= value) return;

    int _expected = barney::__float_as_int(old);
    int _desired  = barney::__float_as_int(value);
    while (true) {
      uint32_t _found = barney::atomicCAS((int*)addr,_expected,_desired);
      if (_found == _expected)
        // write went though; we _did_ write the new mininm and
        // are done.
        return;
      // '_expected' changed, so write did not go through, and
      // somebody else wrote something new to that location.
      old = barney::__int_as_float(_found);
      if (old <= value)
        // somebody else wrote something that's already smaller
        // than what we have ... leave it be, and done.
        return;
      else {
        // somebody else wrote something, but ours is _still_ smaller.
        _expected = _found;
        continue;
      }
    } 
  }

  inline __both__
  void fatomicMax(float *addr, float value)
  {
    float old = *(volatile float *)addr;
    if(old >= value) return;

    int _expected = barney::__float_as_int(old);
    int _desired  = barney::__float_as_int(value);
    while (true) {
      uint32_t _found = barney::atomicCAS((int*)addr,_expected,_desired);
      if (_found == _expected)
        // write went though; we _did_ write the new mininm and
        // are done.
        return;
      // '_expected' changed, so write did not go through, and
      // somebody else wrote something new to that location.
      old = barney::__int_as_float(_found);
      if (old >= value)
        // somebody else wrote something that's already smaller
        // than what we have ... leave it be, and done.
        return;
      else {
        // somebody else wrote something, but ours is _still_ smaller.
        _expected = _found;
        continue;
      }
    } 
  }
  


  
#if BARNEY_BACKEND_EMBREE
  namespace embree {

    __both__ float tex2D1f(barney::rtc::device::TextureObject to,
                           float x, float y);
    __both__ float tex3D1f(barney::rtc::device::TextureObject to,
                           float x, float y, float z);
    __both__ vec4f tex2D4f(barney::rtc::device::TextureObject to,
                           float x, float y);
    __both__ vec4f tex3D4f(barney::rtc::device::TextureObject to,
                           float x, float y, float z);

    inline uint32_t __float_as_int(float f)
    {
      return (const uint32_t &)f;
    }
    inline float __int_as_float(int i)
    {
      return (const float&)i;
    }
    
    struct InstanceGroup;
    struct TraceInterface {
      // #  ifdef __CUDA_ARCH__
      //       /* the embree compute interface only makes sense on the host,
      //          and for the device sometimes isn't even callable (std::atomic
      //          etc), so let's make those fcts 'go away' for device code */
      // #  else
      __both__ void ignoreIntersection(); 
      __both__ void reportIntersection(float t, int i);
      __both__ void *getPRD() const;
      __both__ const void *getProgramData() const;
      __both__ const void *getLPData() const;
      __both__ vec3i getLaunchDims()  const;
      __both__ vec3i getLaunchIndex() const;
      __both__ vec2f getTriangleBarycentrics() const;
      __both__ int getPrimitiveIndex() const;
      __both__ float getRayTmax() const;
      __both__ float getRayTmin() const;
      __both__ vec3f getObjectRayDirection() const;
      __both__ vec3f getObjectRayOrigin() const;
      __both__ vec3f getWorldRayDirection() const;
      __both__ vec3f getWorldRayOrigin() const;
      __both__ vec3f transformNormalFromObjectToWorldSpace(vec3f v) const;
      __both__ vec3f transformPointFromObjectToWorldSpace(vec3f v) const;
      __both__ vec3f transformVectorFromObjectToWorldSpace(vec3f v) const;
      __both__ vec3f transformNormalFromWorldToObjectSpace(vec3f v) const;
      __both__ vec3f transformPointFromWorldToObjectSpace(vec3f v) const;
      __both__ vec3f transformVectorFromWorldToObjectSpace(vec3f v) const;
      __both__ void  traceRay(rtc::device::AccelHandle world,
                              vec3f org,
                              vec3f dir,
                              float t0,
                              float t1,
                              void *prdPtr);

      /* this HAS to be the first entry! :*/
      RTCRayQueryContext embreeRayQueryContext;
      vec3i     launchIndex;
      vec3i     launchDimensions;
      bool      ignoreThisHit;
      float     isec_t;
      vec2f     triangleBarycentrics;
      int       primID;
      int       geomID;
      int       instID;
      vec3f     worldOrigin;
      vec3f     worldDirection;
      void           *prd;
      const void     *geomData;
      const void     *lpData;
      const affine3f *objectToWorldXfm;
      const affine3f *worldToObjectXfm;
      RTCRay         *embreeRay;
      RTCHit         *embreeHit;
      InstanceGroup  *world;
    };
    
    struct ComputeInterface {
      inline __both__ vec3ui launchIndex() const
      {
        return getThreadIdx() + getBlockIdx() * getBlockDim();
      }
#  ifdef __CUDA_ARCH__
      /* the embree compute interface only makes sense on the host,
         and for the device sometimes isn't even callable (std::atomic
         etc), so let's make those fcts 'go away' for device code,
         just like we maek cuda built-in 'threadIdx' etc go away on
         host. */
      inline __both__ vec3ui getThreadIdx() const
      { return vec3ui(0); }
      inline __both__ vec3ui getBlockDim() const
      { return vec3ui(0); }
      inline __both__ vec3ui getBlockIdx() const
      { return vec3ui(0); }
      inline __both__ int atomicAdd(int *ptr, int inc) const
      { return 0; }
      inline __both__ float atomicAdd(float *ptr, float inc) const
      { return 0.f; }
#  else
      inline __both__ vec3ui getThreadIdx() const
      { return this->threadIdx; }
      
      inline __both__ vec3ui getBlockDim() const
      { return this->blockDim; }
      
      inline __both__ vec3ui getBlockIdx() const
      { return this->blockIdx; }
      
      inline __both__ int atomicAdd(int *ptr, int inc) const
      { return ((std::atomic<int> *)ptr)->fetch_add(inc); }
      
      inline __both__ float atomicAdd(float *ptr, float inc) const
      { return ((std::atomic<float> *)ptr)->fetch_add(inc); }
#  endif
      vec3ui threadIdx;
      vec3ui blockIdx;
      vec3ui blockDim;
      vec3ui gridDim;
    };
  }
#endif
  
#ifdef __CUDACC__
  namespace cuda {

    struct ComputeInterface
    {
      inline __device__ vec3ui launchIndex() const
      {
        return getThreadIdx() + getBlockIdx() * getBlockDim();
      }
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

#if 0
      inline __both__ void atomicMin(float *addr, float value) const
      {
        float old = *addr;
        if(old <= value) return;

        int _expected = __float_as_int(old);
        int _desired  = __float_as_int(value);
        while (true) {
          uint32_t _found = ::atomicCAS((int*)addr,_expected,_desired);
          if (_found == _expected)
            // write went though; we _did_ write the new mininm and
            // are done.
            return;
          // '_expected' changed, so write did not go through, and
          // somebody else wrote something new to that location.
          old = __int_as_float(_found);
          if (old <= value)
            // somebody else wrote something that's already smaller
            // than what we have ... leave it be, and done.
            return;
          else {
            // somebody else wrote something, but ours is _still_ smaller.
            _expected = _found;
            continue;
          }
        } 
      }
      inline __both__ void atomicMax(float *addr, float value) const
      {
        float old = *addr;
        if(old >= value) return;

        int _expected = __float_as_int(old);
        int _desired  = __float_as_int(value);
        while (true) {
          uint32_t _found = ::atomicCAS((int*)addr,_expected,_desired);
          if (_found == _expected)
            // write went though; we _did_ write the new mininm and
            // are done.
            return;
          // '_expected' changed, so write did not go through, and
          // somebody else wrote something new to that location.
          old = __int_as_float(_found);
          if (old >= value)
            // somebody else wrote something that's already smaller
            // than what we have ... leave it be, and done.
            return;
          else {
            // somebody else wrote something, but ours is _still_ smaller.
            _expected = _found;
            continue;
          }
        } 
      }
#endif
    };
    
    template<typename KernelT>
    __global__ void
    runKernel(KernelT dd)
    {
      dd.run(::barney::cuda::ComputeInterface());
    }
    
#  define RTC_DECLARE_CUDA_COMPUTE(KernelName,ClassName)                \
    extern "C"                                                          \
    BARNEY_VISIBILITY_DEFAULT                                           \
    void barney_rtc_cuda_launch_##KernelName(::barney::vec3ui nb,       \
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
#else
#  define RTC_DECLARE_CUDA_COMPUTE(KernelName,ClassName)  /* nothing */
#endif
  
#if BARNEY_BACKEND_EMBREE
#  define RTC_DECLARE_EMBREE_COMPUTE(KernelName,ClassName)              \
  extern "C"                                                            \
  BARNEY_VISIBILITY_DEFAULT                                             \
  void barney_rtc_embree_computeBlock_##KernelName                      \
  (::barney::embree::ComputeInterface &ci,const void *dd)               \
  {                                                                     \
    auto &tid = ci.threadIdx;                                           \
    for (tid.z=0;tid.z<ci.blockDim.z;tid.z++)                           \
      for (tid.y=0;tid.y<ci.blockDim.y;tid.y++)                         \
        for (tid.x=0;tid.x<ci.blockDim.x;tid.x++)                       \
          ((ClassName*)dd)->run(ci);                                    \
  }
  
#  define RTC_DECLARE_EMBREE_TRACE(KernelName,RayGenType)               \
  extern "C"                                                            \
  BARNEY_VISIBILITY_DEFAULT                                             \
  void barney_rtc_embree_trace_##KernelName                             \
  (::barney::embree::TraceInterface &rt)                                \
  {                                                                     \
    RayGenType::run(rt);                                                \
  }

#define RTC_DECLARE_EMBREE_USER_GEOM(name,type)             \
                                                            \
  extern "C"                                                \
  BARNEY_VISIBILITY_DEFAULT                                 \
  void barney_embree_ch_##name                              \
  (::barney::embree::TraceInterface &rtcore)                \
  { type::closest_hit(rtcore); }                            \
                                                            \
  extern "C"                                                \
  BARNEY_VISIBILITY_DEFAULT                                 \
  void barney_embree_ah_##name                              \
  (::barney::embree::TraceInterface &rtcore)                \
  { type::any_hit(rtcore); }                                \
                                                            \
  extern "C"                                                \
  BARNEY_VISIBILITY_DEFAULT                                 \
  void barney_embree_bounds_##name                          \
  (::barney::embree::TraceInterface &rtcore,                \
   const void *dd,                                          \
   ::owl::common::box3f &bounds,                            \
   int primID)                                              \
  { type::bounds(rtcore,dd,bounds,primID); }                \
                                                            \
  extern "C"                                                \
  BARNEY_VISIBILITY_DEFAULT                                 \
  void barney_embree_intersect_##name                       \
  (::barney::embree::TraceInterface &rtcore)                \
  { type::intersect(rtcore); }                               
  
  
#define RTC_DECLARE_EMBREE_TRIANGLES_GEOM(name,type)        \
                                                            \
  extern "C"                                                \
  BARNEY_VISIBILITY_DEFAULT                                 \
  void barney_embree_ch_##name                              \
  (::barney::embree::TraceInterface &rtcore)                \
  { type::closest_hit(rtcore); }                            \
                                                            \
  extern "C"                                                \
  BARNEY_VISIBILITY_DEFAULT                                 \
  void barney_embree_ah_##name                              \
  (::barney::embree::TraceInterface &rtcore)                \
  { type::any_hit(rtcore); }                               
  
#else
#  define RTC_DECLARE_EMBREE_COMPUTE(KernelName,ClassName) /* nothing */
#  define RTC_DECLARE_EMBREE_TRACE(KernelName,ClassName) /* nothing */
#  define RTC_DECLARE_EMBREE_TRIANGLES_GEOM(name,type) /* nothing */
#  define RTC_DECLARE_EMBREE_USER_GEOM(name,type)        /* nothing */
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
      
      
      inline __device__ void traceRay(rtc::device::AccelHandle world,
                                      vec3f org,
                                      vec3f dir,
                                      float t0,
                                      float t1,
                                      void *prdPtr) 
      {
        unsigned int           p0 = 0;
        unsigned int           p1 = 0;
        owl::packPointer(prdPtr,p0,p1);
        
        uint32_t rayFlags = 0u;
        owl::Ray ray(org,dir,t0,t1);
        optixTrace((const OptixTraversableHandle &)world,
                   (const ::float3&)ray.origin,
                   (const ::float3&)ray.direction,
                   ray.tmin,
                   ray.tmax,
                   ray.time,
                   ray.visibilityMask,
                   /*rayFlags     */ rayFlags,
                   /*SBToffset    */ ray.rayType,
                   /*SBTstride    */ ray.numRayTypes,
                   /*missSBTIndex */ ray.rayType,              
                   p0,
                   p1);
      }
    };

#define RTC_DECLARE_GLOBALS(Type)                               \
    __constant__ Type optixLaunchParams;                        \
    namespace barney {                                          \
      namespace optix {                                         \
        inline __device__ const void *getLaunchParamsPointer()  \
        { return &optixLaunchParams; }                          \
      }                                                         \
    }

# define RTC_DECLARE_OPTIX_TRACE(name,RayGenType)          \
    extern "C"  __global__                                       \
    void __raygen__##name()                                     \
    {                                                           \
      RayGenType *rg = (RayGenType*)optixGetSbtDataPointer();   \
      ::barney::optix::RTCoreInterface rtcore;                  \
      rg->run(rtcore);                                          \
    }                                                                   
    

#define RTC_DECLARE_OPTIX_USER_GEOM(name,type)                  \
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
    void __intersection__##name() {                             \
      ::barney::optix::RTCoreInterface rtcore;                  \
      type::intersect(rtcore);                                  \
    }                                                           \
                                                                \
    __device__ void __boundsFunc__##name(const void *geom,              \
                                         owl::common::box3f &result,    \
                                         int primID)                    \
    {                                                                   \
      ::barney::optix::RTCoreInterface rtcore;                          \
      type::bounds(rtcore,geom,result,primID);                          \
    }                                                                   \
    extern "C" __global__                                               \
    void __boundsFuncKernel__##name(const void *geom,                   \
                                    owl::common::box3f *boundsArray,    \
                                    int numPrims)                       \
    {                                                                   \
      uint32_t blockIndex                                               \
        = blockIdx.x                                                    \
        + blockIdx.y * gridDim.x                                        \
        + blockIdx.z * gridDim.x * gridDim.y;                           \
      uint32_t primID                                                   \
        = threadIdx.x + blockDim.x*threadIdx.y                          \
        + blockDim.x*blockDim.y*blockIndex;                             \
      if (primID < numPrims) {                                          \
        __boundsFunc__##name(geom,boundsArray[primID],primID);          \
      }                                                                 \
    }                                                                   \
                                                                        \
    
  
#define RTC_DECLARE_OPTIX_TRIANGLES_GEOM(name,type)     \
                                                        \
    extern "C" __global__                               \
    void __closesthit__##name() {                       \
      ::barney::optix::RTCoreInterface rtcore;          \
      type::closest_hit(rtcore);                        \
    }                                                   \
                                                        \
    extern "C" __global__                               \
    void __anyhit__##name() {                           \
      ::barney::optix::RTCoreInterface rtcore;          \
      type::any_hit(rtcore);                            \
    }                                                   \
  
  } // ::barney::optix
#else
# define RTC_DECLARE_GLOBALS(Type) /* nothing */
# define RTC_DECLARE_OPTIX_TRIANGLES_GEOM(name,type) /* nothing */
# define RTC_DECLARE_OPTIX_USER_GEOM(name,type) /* nothing */
# define RTC_DECLARE_OPTIX_TRACE(name,RayGenType)                       \
  extern char name##_ptx[];                                             \
  extern "C" BARNEY_VISIBILITY_DEFAULT                                  \
  char *get_ptx_##name() { return name##_ptx; }                           

  
#endif  
}

namespace barney {
  namespace rtc {
    // TODO:
    template<typename T>
    inline __both__ T tex1D(barney::rtc::device::TextureObject to,
                                       float x)
    {
#ifdef __CUDA_ARCH__
      printf("tex1d missing...\n");
      return T{};
#else
      BARNEY_NYI();
#endif
    }
    template<typename T>
    inline __both__ T tex2D(barney::rtc::device::TextureObject to,
                            float x, float y);


    template<>
    inline __both__ vec4f tex2D<vec4f>(barney::rtc::device::TextureObject to,
                                         float x, float y)
    {
#ifdef __CUDA_ARCH__
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      ::float4 v = ::tex2D<::float4>(texObj,x,y);
      return (const vec4f &)v;
      // return T{};
#elif BARNEY_BACKEND_EMBREE
      // this in on th ehost, and we _do_ have the embree backend built in:
      return embree::tex2D4f(to,x,y);
#else
      // this cannot possibly happen because we have to have either a
      // cuda or an embree backend to even call this.
      return {0.f,0.f,0.f,0.f};
#endif
    }


    template<>
    inline __both__ float tex2D<float>(barney::rtc::device::TextureObject to,
                                       float x, float y)
    {
#ifdef __CUDA_ARCH__
      // we _must_ be on the device, so this is a cuda teture
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      return ::tex2D<float>(texObj,x,y);
      // return T{};
#elif BARNEY_BACKEND_EMBREE
      // this in on th ehost, and we _do_ have the embree backend built in:
      return embree::tex2D1f(to,x,y);
#else
      // this cannot possibly happen because we have to have either a
      // cuda or an embree backend to even call this.
      return 0.f;
#endif
    }

    template<typename T>
    inline __both__ T tex3D(barney::rtc::device::TextureObject to,
                                       float x, float y, float z);
    
    template<>
    inline __both__
    float tex3D<float>(barney::rtc::device::TextureObject to,
                       float x, float y, float z)
    {
#ifdef __CUDA_ARCH__
      // we _must_ be on the device, so this is a cuda teture
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      float f= ::tex3D<float>(texObj,x,y,z);
      // printf("tex3d -> %f\n",f);
      return f;
      // return T{};
#elif BARNEY_BACKEND_EMBREE
      // this in on th ehost, and we _do_ have the embree backend built in:
      return embree::tex3D1f(to,x,y,z);
#else
      // this cannot possibly happen because we have to have either a
      // cuda or an embree backend to even call this.
      return 0.f;
#endif
    }


    template<>
    inline __both__
    vec4f tex3D<vec4f>(barney::rtc::device::TextureObject to,
                       float x, float y, float z)
    {
#ifdef __CUDA_ARCH__
      // we _must_ be on the device, so this is a cuda teture
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      ::float4 v = ::tex3D<::float4>(texObj,x,y,z);
      return load(v);
      // return T{};
#elif BARNEY_BACKEND_EMBREE
      // this in on the host, and we _do_ have the embree backend built in:
      return embree::tex3D4f(to,x,y,z);
#else
      // this cannot possibly happen because we have to have either a
      // cuda or an embree backend to even call this.
      return 0.f;
#endif
    }
  }
}


#  define RTC_DECLARE_COMPUTE(name,kernel)         \
  RTC_DECLARE_CUDA_COMPUTE(name,kernel)            \
  RTC_DECLARE_EMBREE_COMPUTE(name,kernel)

#  define RTC_DECLARE_TRACE(name,kernel)         \
  RTC_DECLARE_OPTIX_TRACE(name,kernel)           \
  RTC_DECLARE_EMBREE_TRACE(name,kernel)

#define RTC_DECLARE_TRIANGLES_GEOM(name,type)   \
  RTC_DECLARE_OPTIX_TRIANGLES_GEOM(name,type)   \
  RTC_DECLARE_EMBREE_TRIANGLES_GEOM(name,type)     


#define RTC_DECLARE_USER_GEOM(name,type)   \
  RTC_DECLARE_OPTIX_USER_GEOM(name,type)   \
  RTC_DECLARE_EMBREE_USER_GEOM(name,type)     





#endif
