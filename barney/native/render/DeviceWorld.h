// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/light/EnvMap.h"
#include "native/light/DirLight.h"
#include "native/light/PointLight.h"
#include "native/light/QuadLight.h"

namespace BARNEY_NS {
  namespace native {
    
    struct DeviceWorld {
      int                  numQuadLights = 0;
      const QuadLight::DD *quadLights    = nullptr;
      int                  numDirLights  = 0;
      const DirLight::DD  *dirLights     = nullptr;
      int                  numPointLights  = 0;
      const PointLight::DD  *pointLights     = nullptr;
      int                 *instIDToUserInstID = 0;
      // Indexed by native input instance ID, like instance attributes.
      const float         *instanceOrientations = nullptr;
      const affine3f      *motionDeltas       = nullptr;
      int                  numMotionDeltas    = 0;

      const DeviceMaterial *materials;
      const Sampler::DD    *samplers;
      const rtc::float4    *instanceAttributes[5];
      EnvMapLight::DD       envMapLight;
      // uint32_t              rngSeed;
      uint32_t              rank;
    };
    
  }
}
