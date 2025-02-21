#pragma once

#include "rtcore/common/rtcore-common.h"

#if BARNEY_RTC_OPTIX
# include "optix/RTCore.h"

namespace rtc {
  using namespace optix;

#if __CUDA_ARCH__
  using cuda_common::load;
  using cuda_common::tex1D;
# define RTC_DEVICE_CODE 1
#endif
}
#endif

