// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/cudaCommon/ComputeInterface.h"

namespace rtc {
  namespace cuda {
    using cuda_common::ComputeInterface;
#if RTC_DEVICE_CODE
    
    using cuda_common::tex1D;
    using cuda_common::tex2D;
    using cuda_common::tex3D;
    
    using cuda_common::fatomicMin;
    using cuda_common::fatomicMax;
#endif
  }
}

# define __rtc_global __global__
# define __rtc_launch(myRTC,kernel,nb,bs,...)                           \
  {                                                                     \
    rtc::cuda::SetActiveGPU forDuration(myRTC);                         \
    if (nb)                                                             \
      kernel<<<nb,bs,0,myRTC->stream>>>                                 \
        (rtc::cuda::ComputeInterface(), __VA_ARGS__);                   \
    PING; BARNEY_CUDA_SYNC_CHECK();                                      \
  }
  
