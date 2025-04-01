#pragma once

#include "rtcore/common/rtcore-common.h"

#if BARNEY_RTC_OPTIX || BARNEY_RTC_CUDA
# if defined(__CUDACC__) || defined(__HIPCC__)
# include "rtcore/cudaCommon/ComputeInterface.h"

namespace rtc {
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  using ::rtc::cuda_common::fatomicMin;
  using ::rtc::cuda_common::fatomicMax;
#endif
  using ::rtc::cuda_common::ComputeInterface;
  using ::rtc::cuda_common::load;
  using ::rtc::cuda_common::tex1D;
  using ::rtc::cuda_common::tex1D;
  using ::rtc::cuda_common::tex2D;
  using ::rtc::cuda_common::tex3D;

# define RTC_DEVICE_CODE 1
}
# endif
#endif

