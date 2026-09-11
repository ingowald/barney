// SPDX-FileCopyrightText: Copyright (c) 2025-2026NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/Object.h"
#include "native/render/Ray.h"

namespace BARNEY_NS {
  namespace native {
    
/*! a kernel that is to be applie to an entire wave of rays;
        allows a geometry (or volume) to _not_ craete an rtc
        (user-)geom, and instead install a kernel that will be
        executed for all rays in the queue, in a cuda kernel */
    struct WholeWaveIntersectKernel {
      typedef std::shared_ptr<WholeWaveIntersectKernel> SP;
      virtual void run(Device *device,
                       const affine3f &instantiationXfm,
                       Ray *d_rays,
                       int numRays) = 0;
    };
    
  }
}
