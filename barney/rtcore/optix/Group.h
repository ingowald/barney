// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/optix/Device.h"
#include <owl/owl.h>

namespace BARNEY_NS {
  namespace rtc {

    struct Device;

    struct Group {
      Group(Device *device, OWLGroup owlGroup);
      virtual ~Group() { owlGroupRelease(owl); }
      
      rtc::AccelHandle getDD() const;
      void buildAccel();
      void refitAccel();
      void setTransforms(const std::vector<affine3f> &xfms);
      
      OWLGroup const owl;
      Device *const device;
    };

  }
}
