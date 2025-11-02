// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/cudaCommon/ComputeInterface.h"

namespace rtc {
  namespace optix {
#ifdef __CUDACC__
    using cuda_common::ComputeInterface;
    
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
  { rtc::optix::SetActiveGPU forDuration(myRTC); if (nb) kernel<<<nb,bs,0,myRTC->stream>>>(rtc::optix::ComputeInterface(), __VA_ARGS__); }
