// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/embree/embree-common.h"
#include "rtcore/embree/Device.h"
#include "rtcore/embree/Buffer.h"
#include "rtcore/embree/Group.h"
#include "rtcore/embree/ComputeKernel.h"
#include "rtcore/embree/Denoiser.h"

namespace rtc {
  namespace embree {
    
    inline bool enablePeerAccess(const std::vector<int> &IDs)
    { /* ignore / no-op on embree backend */; return true; }
    
    /*! get a unique hash for a given physical device. */
    size_t getPhysicalDeviceHash(int gpuID);
    
  }
}

    

