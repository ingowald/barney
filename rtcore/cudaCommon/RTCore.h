DEPRECATED

// #pragma once

// #include <cuda.h>
// #include "Device.h"

// namespace rtc {
//   namespace cuda_common {
// #ifdef __CUDACC__
//     using ::atomicCAS;
//     // using ::__uint_as_float;
//     // using ::__float_as_int;

//     inline __device__
//     void fatomicMin(float *addr, float value)
//     {
//       float old = *(volatile float *)addr;
//       if(old <= value) return;

//       int _expected = __float_as_int(old);
//       int _desired  = __float_as_int(value);
//       while (true) {
//         uint32_t _found = atomicCAS((int*)addr,_expected,_desired);
//         if (_found == _expected)
//           // write went though; we _did_ write the new mininm and
//           // are done.
//           return;
//         // '_expected' changed, so write did not go through, and
//         // somebody else wrote something new to that location.
//         old = __int_as_float(_found);
//         if (old <= value)
//           // somebody else wrote something that's already smaller
//           // than what we have ... leave it be, and done.
//           return;
//         else {
//           // somebody else wrote something, but ours is _still_ smaller.
//           _expected = _found;
//           continue;
//         }
//       } 
//     }

//     inline __device__
//     void fatomicMax(float *addr, float value)
//     {
//       float old = *(volatile float *)addr;
//       if(old >= value) return;

//       int _expected = __float_as_int(old);
//       int _desired  = __float_as_int(value);
//       while (true) {
//         uint32_t _found = atomicCAS((int*)addr,_expected,_desired);
//         if (_found == _expected)
//           // write went though; we _did_ write the new mininm and
//           // are done.
//           return;
//         // '_expected' changed, so write did not go through, and
//         // somebody else wrote something new to that location.
//         old = __int_as_float(_found);
//         if (old >= value)
//           // somebody else wrote something that's already smaller
//           // than what we have ... leave it be, and done.
//           return;
//         else {
//           // somebody else wrote something, but ours is _still_ smaller.
//           _expected = _found;
//           continue;
//         }
//       } 
//     }

    
//     template<typename T>
//     inline __device__ T tex1D(rtc::device::TextureObject to,
//                               float x)
//     { assert(0); }
    
//     template<typename T>
//     inline __device__ T tex2D(rtc::device::TextureObject to,
//                               float x, float y)
//     { assert(0); }
    



//     template<>
//     inline __device__ vec4f tex1D<vec4f>(rtc::device::TextureObject to,
//                                          float x)
//     {
//       cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
//       ::float4 v = ::tex1D<::float4>(texObj,x);
//       return load(v);
//     }
    
    
//     template<>
//     inline __device__ vec4f tex2D<vec4f>(rtc::device::TextureObject to,
//                                          float x, float y)
//     {
//       cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
//       ::float4 v = ::tex2D<::float4>(texObj,x,y);
//       return v;
//     }

//     template<>
//     inline __device__ float tex2D<float>(rtc::device::TextureObject to,
//                                          float x, float y)
//     {
//       cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
//       return ::tex2D<float>(texObj,x,y);
//     }

//     template<typename T>
//     inline __device__ T tex3D(rtc::device::TextureObject to,
//                               float x, float y, float z);
    
//     template<>
//     inline __device__
//     float tex3D<float>(rtc::device::TextureObject to,
//                        float x, float y, float z)
//     {
//       cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
//       float f= ::tex3D<float>(texObj,x,y,z);
//       return f;
//     }

//     template<>
//     inline __device__
//     vec4f tex3D<vec4f>(rtc::device::TextureObject to,
//                        float x, float y, float z)
//     {
//       cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
//       ::float4 v = ::tex3D<::float4>(texObj,x,y,z);
//       return load(v);
//     }

//     struct ComputeInterface
//     {
//       inline __device__ vec3ui launchIndex() const
//       {
//         return getThreadIdx() + getBlockIdx() * getBlockDim();
//       }
//       inline __device__ vec3ui getThreadIdx() const
//       { return threadIdx; }
//       inline __device__ vec3ui getBlockDim() const
//       { return {blockDim.x,blockDim.y,blockDim.z}; }
//       inline __device__ vec3ui getBlockIdx() const
//       { return {blockIdx.x,blockIdx.y,blockIdx.z}; }
//       inline __device__ int atomicAdd(int *ptr, int inc) const
//       { return ::atomicAdd(ptr,inc); }
//       inline __device__ float atomicAdd(float *ptr, float inc) const
//       { return ::atomicAdd(ptr,inc); }
//     };
// #endif
//   }
// }


