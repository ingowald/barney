#pragma once

#if BARNEY_RTC_OPTIX
# include "optix/Device.h"
# include "optix/Geom.h" 
# include "optix/Group.h" 
# include "optix/Denoiser.h" 
# include "cuda/CUDACommon.h" 
# include "cuda/RTCore.h" 

namespace rtc {
  using namespace optix;
  
  using rtc::cuda_common::ComputeKernel1D;
  using rtc::cuda_common::ComputeKernel2D;
  using rtc::cuda_common::ComputeKernel3D;
  using rtc::cuda_common::load;

#ifdef __CUDACC__
# define RTC_DEVICE_CODE 1
#endif
  
}
#endif
