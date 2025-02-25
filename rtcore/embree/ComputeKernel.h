#pragma once

#include "rtcore/embree/Device.h"
#include "rtcore/embree/ComputeInterface.h"

namespace rtc {
  namespace embree {

    struct ComputeKernel1D {
      void (*launch)(unsigned int nb, unsigned int bs,
                     const void *pKernelData) = 0;
      // virtual void launch(unsigned int nb, unsigned int bs,
      //                     const void *pKernelData) = 0;
    };
    struct ComputeKernel2D {
      void (*launch)(vec2ui nb, vec2ui bs,
                     const void *pKernelData) = 0;
      // virtual void launch(vec2ui nb, vec2ui bs,
      //                     const void *pKernelData) = 0;
      // inline void launch(vec2i nb, vec2i bs,
      //                    const void *pKernelData)
      // { launch(vec2ui(nb),vec2ui(bs),pKernelData); }
    };
    struct ComputeKernel3D {
      void (*launch)(vec3ui nb, vec3ui bs,
                     const void *pKernelData) = 0;
      // virtual void launch(vec3ui nb, vec3ui bs,
      //                     const void *pKernelData) = 0;
      // inline void launch(vec3i nb, vec3i bs,
      //                    const void *pKernelData)
      // { launch(vec3ui(nb),vec3ui(bs),pKernelData); }
    };


  }
}

#define RTC_EXPORT_COMPUTE1D(name,ClassName)                            \
  void rtc_embree_compute_##name(unsigned nb, unsigned bs,              \
                                 const void *pData)                     \
  {                                                                     \
    rtc::embree::ComputeInterface ci;                                   \
    ((ClassName *)pData)->run(ci);                                      \
  }                                                                     \
                                                                        \
  rtc::ComputeKernel1D *createCompute_##name(rtc::Device *dev)          \
  { return new rtc::ComputeKernel1D{rtc_embree_compute_##name}; }       \
                                                               
#define RTC_EXPORT_COMPUTE2D(name,ClassName)                            \
  void rtc_embree_compute_##name(rtc::vec2ui nb, rtc::vec2ui bs,             \
                                 const void *pData)                     \
  {                                                                     \
    rtc::embree::ComputeInterface ci;                                   \
    ((ClassName *)pData)->run(ci);                                      \
  }                                                                     \
                                                                        \
  rtc::ComputeKernel2D *createCompute_##name(rtc::Device *dev)          \
  { return new rtc::ComputeKernel2D{rtc_embree_compute_##name}; }       \
                                                               
#define RTC_EXPORT_COMPUTE3D(name,ClassName)                            \
  void rtc_embree_compute_##name(rtc::vec3ui nb, rtc::vec3ui bs,              \
                                 const void *pData)                     \
  {                                                                     \
    rtc::embree::ComputeInterface ci;                                   \
    ((ClassName *)pData)->run(ci);                                      \
  }                                                                     \
                                                                        \
  rtc::ComputeKernel3D *createCompute_##name(rtc::Device *dev)          \
  { return new rtc::ComputeKernel3D{rtc_embree_compute_##name}; }       \
                                                               



#define RTC_IMPORT_COMPUTE1D(name)                            \
  extern rtc::ComputeKernel1D *createCompute_##name(rtc::Device *dev);

#define RTC_IMPORT_COMPUTE2D(name)                            \
  extern rtc::ComputeKernel2D *createCompute_##name(rtc::Device *dev);

#define RTC_IMPORT_COMPUTE3D(name)                            \
  extern rtc::ComputeKernel3D *createCompute_##name(rtc::Device *dev);

