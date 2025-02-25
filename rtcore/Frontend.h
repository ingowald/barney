#pragma once

#if BARNEY_RTC_OPTIX
# include "optix/Device.h"
# include "optix/Geom.h" 
# include "optix/Group.h" 
# include "optix/Denoiser.h" 
# include "cuda/CUDACommon.h" 
# include "cuda/RTCore.h" 
# if BARNEY_DEVICE_PROGRAM
#  include "optix/TraceInterface.h"
# endif

namespace rtc {
  using namespace optix;
  
  using rtc::cuda_common::ComputeKernel1D;
  using rtc::cuda_common::ComputeKernel2D;
  using rtc::cuda_common::ComputeKernel3D;

  using rtc::cuda_common::float4;
  using rtc::cuda_common::load;

# ifdef __CUDACC__
#  define RTC_DEVICE_CODE 1
# endif
# if !BARNEY_DEVICE_PROGRAM
  struct TraceInterface;
# endif
}
#endif



#if BARNEY_RTC_EMBREE
# include "rtcore/embree/Device.h"
# include "rtcore/embree/Geom.h" 
# include "rtcore/embree/GeomType.h" 
# include "rtcore/embree/Group.h" 
# include "rtcore/embree/Denoiser.h" 
# include "rtcore/embree/ComputeInterface.h" 
# include "rtcore/embree/ComputeKernel.h"
# if BARNEY_DEVICE_PROGRAM
#  include "rtcore/embree/TraceInterface.h"
# endif

namespace rtc {
  using namespace rtc::embree;

  using rtc::embree::ComputeInterface;
  using rtc::embree::ComputeKernel1D;
  using rtc::embree::ComputeKernel2D;
  using rtc::embree::ComputeKernel3D;
  using rtc::embree::load;
  using rtc::embree::tex1D;
  using rtc::embree::tex2D;
  using rtc::embree::tex3D;

# define RTC_DEVICE_CODE 1
  
# if !BARNEY_DEVICE_PROGRAM
  struct TraceInterface;
# endif
}
#endif
