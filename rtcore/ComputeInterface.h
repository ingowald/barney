// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


/* kernelInterface - how device-side compute kernels (e.g., shadeRays,
   but not programs like intersect or anyhit) can talk to rtcore */

#pragma once

#if BARNEY_RTC_CUDA
# include "cuda/ComputeInterface.h"
namespace rtc { using namespace rtc::cuda; }
#endif

#if BARNEY_RTC_OPTIX
# include "optix/ComputeInterface.h"
namespace rtc { using namespace rtc::optix; }
#endif

#if BARNEY_RTC_EMBREE
# include "embree/ComputeInterface.h"
namespace rtc { using namespace rtc::embree; }
#endif
