#pragma once

#include "rtcore/embree/Device.h"
#include "rtcore/embree/Buffer.h"
#include "rtcore/embree/Group.h"
#include "rtcore/embree/ComputeKernel.h"
#include "rtcore/embree/Denoiser.h"

namespace rtc {
  namespace embree {
    
    inline bool enablePeerAccess(const std::vector<int> &IDs)
    { /* ignore / no-op on embree backend */; return true; }

  }
}

    

