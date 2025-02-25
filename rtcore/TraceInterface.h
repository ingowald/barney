#pragma once

#if BARNEY_RTC_OPTIX
# include "optix/TraceInterface.h"
namespace rtc {
  using rtc::optix::TraceInterface;
}
#endif



#if BARNEY_RTC_EMBREE
# include "embree/TraceInterface.h"
namespace rtc {
  using rtc::embree::TraceInterface;
}
#endif


