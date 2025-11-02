// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/embree/Device.h"
#include "rtcore/embree/ComputeInterface.h"

namespace rtc {
  namespace embree {

    struct ComputeKernel1D {
      // void (*launch)(unsigned int nb, unsigned int bs,
      //                const void *pKernelData) = 0;
      void launch(unsigned int nb, unsigned int bs,
                  const void *pKernelData);
      Device *device;
      void (*computeFct)(rtc::embree::ComputeInterface &ci,
                        const void *pKernelData) = 0;
    };
    struct ComputeKernel2D {
      // void (*launch)(vec2ui nb, vec2ui bs,
      //                const void *pKernelData) = 0;
      void launch(vec2ui nb, vec2ui bs,
                  const void *pKernelData);
      // inline void launch(vec2i nb, vec2i bs,
      //                    const void *pKernelData)
      // { launch(vec2ui(nb),vec2ui(bs),pKernelData); }
      Device *device;
      void (*computeFct)(rtc::embree::ComputeInterface &ci,
                        const void *pKernelData) = 0;
    };
    struct ComputeKernel3D {
      // void (*launch)(vec3ui nb, vec3ui bs,
      //                const void *pKernelData) = 0;
      void launch(vec3ui nb, vec3ui bs,
                  const void *pKernelData);
      // inline void launch(vec3i nb, vec3i bs,
      //                    const void *pKernelData)
      // { launch(vec3ui(nb),vec3ui(bs),pKernelData); }
      Device *device;
      void (*computeFct)(rtc::embree::ComputeInterface &ci,
                        const void *pKernelData) = 0;
    };


  }
}

#define RTC_EXPORT_COMPUTE1D(name,ClassName)                            \
  void rtc_embree_compute_##name(rtc::embree::ComputeInterface &ci,     \
                                 const void *pData)                     \
  {                                                                     \
    ((ClassName *)pData)->run(ci);                                      \
  }                                                                     \
                                                                        \
  rtc::ComputeKernel1D *createCompute_##name(rtc::Device *dev)          \
  { return new rtc::ComputeKernel1D{dev,rtc_embree_compute_##name}; }   \
  
#define RTC_EXPORT_COMPUTE2D(name,ClassName)                            \
  void rtc_embree_compute_##name(rtc::embree::ComputeInterface &ci,     \
                                 const void *pData)                     \
  {                                                                     \
    ((ClassName *)pData)->run(ci);                                      \
  }                                                                     \
                                                                        \
  rtc::ComputeKernel2D *createCompute_##name(rtc::Device *dev)          \
  { return new rtc::ComputeKernel2D{dev,rtc_embree_compute_##name}; }   \
                                                               
#define RTC_EXPORT_COMPUTE3D(name,ClassName)                            \
  void rtc_embree_compute_##name(rtc::embree::ComputeInterface &ci,     \
                                 const void *pData)                     \
  {                                                                     \
    ((ClassName *)pData)->run(ci);                                      \
  }                                                                     \
                                                                        \
  rtc::ComputeKernel3D *createCompute_##name(rtc::Device *dev)          \
  { return new rtc::ComputeKernel3D{dev,rtc_embree_compute_##name}; }   \
  



#define RTC_IMPORT_COMPUTE1D(name)                            \
  extern rtc::ComputeKernel1D *createCompute_##name(rtc::Device *dev);

#define RTC_IMPORT_COMPUTE2D(name)                            \
  extern rtc::ComputeKernel2D *createCompute_##name(rtc::Device *dev);

#define RTC_IMPORT_COMPUTE3D(name)                            \
  extern rtc::ComputeKernel3D *createCompute_##name(rtc::Device *dev);

