// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


/*! \file rtcore/cudaCommon/ComputeKerne.h Defines basic abstraction
  for 1D, 2D, and 3D compute kernels, and the IMPORT macros to
  import such kernels on host code */

#pragma once

#include "rtcore/cudaCommon/cuda-common.h"

namespace rtc {
  namespace cuda_common {

    struct ComputeKernel1D {
      virtual ~ComputeKernel1D() = default;
      virtual void launch(unsigned int nb, unsigned int bs,
                          const void *pKernelData) = 0;
    };
    
    struct ComputeKernel2D {
      virtual ~ComputeKernel2D() = default;
      virtual void launch(vec2ui nb, vec2ui bs,
                          const void *pKernelData) = 0;
    };
    
    struct ComputeKernel3D {
      virtual ~ComputeKernel3D() = default;
      virtual void launch(vec3ui nb, vec3ui bs,
                          const void *pKernelData) = 0;
    };
    
  }
}

#define RTC_IMPORT_COMPUTE1D(name)                                      \
    rtc::ComputeKernel1D *createCompute_##name(rtc::Device *dev);       
#define RTC_IMPORT_COMPUTE2D(name)                                      \
    rtc::ComputeKernel2D *createCompute_##name(rtc::Device *dev);       
#define RTC_IMPORT_COMPUTE3D(name)                                      \
    rtc::ComputeKernel3D *createCompute_##name(rtc::Device *dev);       


