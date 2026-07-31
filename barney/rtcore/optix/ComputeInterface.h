// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/cudaCommon/ComputeInterface.h"

namespace BARNEY_NS {
  namespace rtc {

// #ifdef __CUDACC__
//     using cuda_common::ComputeInterface;
    
//     using cuda_common::tex1D;
//     using cuda_common::tex2D;
//     using cuda_common::tex3D;
    
//     using cuda_common::fatomicMin;
//     using cuda_common::fatomicMax;
// #endif
  }
}

// # define __rtc_global __global__
// # define __rtc_launch(myRTC,kernel,nb,bs,...)                           \
//   {                                                                     \
//     ::BARNEY_NS::rtc::SetActiveGPU forDuration(myRTC);                  \
//     if (nb) kernel<<<nb,bs,0,myRTC->stream>>>                           \
//               (::BARNEY_NS::rtc::ComputeInterface(), __VA_ARGS__);      \
//   }
