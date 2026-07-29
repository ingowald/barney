// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/cudaCommon/ComputeInterface.h"

// # define __rtc_global __global__
// # define __rtc_launch(myRTC,kernel,nb,bs,...)                           \
//   {                                                                     \
//     rtc::cuda::SetActiveGPU forDuration(myRTC);                         \
//     if (nb)                                                             \
//       kernel<<<nb,bs,0,myRTC->stream>>>                                 \
//         (rtc::cuda::ComputeInterface(), __VA_ARGS__);                   \
//   }

