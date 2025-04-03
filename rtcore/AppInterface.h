#pragma once

#if BARNEY_RTC_CUDA
# include "cuda/AppInterface.h"
namespace rtc { using namespace rtc::cuda; }
#endif
