// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/optix/Device.h"
#include <owl/owl.h>

namespace rtc {
  namespace optix {
    struct Device;

    struct Group {
      Group(optix::Device *device, OWLGroup owlGroup);
      virtual ~Group() { owlGroupRelease(owl); }
      
      rtc::AccelHandle getDD() const;
      void buildAccel();
      void refitAccel();
      
      OWLGroup const owl;
      optix::Device *const device;
    };

  }
}
