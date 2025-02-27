#pragma once

#include "rtcore/common/rtcore-common.h"

#if BARNEY_RTC_OPTIX
# include "rtcore/cudaCommon/ComputeInterface.h"

namespace rtc {
#ifdef __CUDACC__
  using rtc::cuda_common::fatomicMin;
  using rtc::cuda_common::fatomicMax;
  using rtc::cuda_common::ComputeInterface;
  using rtc::cuda_common::load;
  using rtc::cuda_common::tex1D;
  using rtc::cuda_common::tex1D;
  using rtc::cuda_common::tex2D;
  using rtc::cuda_common::tex3D;

# define RTC_DEVICE_CODE 1
#endif
}
#endif

