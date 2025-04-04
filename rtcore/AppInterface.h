#pragma once

#if BARNEY_RTC_CUDA
# include "cuda/AppInterface.h"
namespace rtc { using namespace rtc::cuda; }
#endif

#if BARNEY_RTC_OPTIX
# include "optix/AppInterface.h"
namespace rtc { using namespace rtc::optix; }
#endif

#if BARNEY_RTC_EMBREE
# include "embree/AppInterface.h"
namespace rtc { using namespace rtc::embree; }
#endif
