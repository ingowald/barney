
/* kernelInterface - how device-side compute kernels (e.g., shadeRays,
   but not programs like intersect or anyhit) can talk to rtcore */

#pragma once

#if BARNEY_RTC_CUDA
# include "cuda/ComputeInterface.h"
namespace rtc { using namespace rtc::cuda; }
#endif
