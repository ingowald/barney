#pragma once

#if BARNEY_RTC_OPTIX
# include "optix/Device.h"
# include "optix/Geom.h" 
# include "optix/Group.h" 
# include "cuda/CUDACommon.h" 

namespace rtc {
  using namespace optix;
  
  using cuda_common::load;
  
}
#endif
