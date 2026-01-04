// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/DeviceGroup.h"
#include "barney/render/World.h"

namespace BARNEY_NS {

  namespace render {
    struct Ray;
  };
  
  struct MultiPassObject {
    typedef std::shared_ptr<MultiPassObject> SP;
    virtual void launch(Device *device,
                        const render::World::DD &world,
                        const affine3f &instanceXfm,
                        render::Ray *rays,
                        int numRays) = 0;
  };
  
}
