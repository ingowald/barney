// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/native/common/barney-common.h"
#include "native/volume/TransferFunction.h"

namespace BARNEY_NS {
  namespace native {
    
    struct Device;

    /*! Direct two-transfer-function volume model. The regular transfer
      function supplies scattering tint and extinction; the emission transfer
      function supplies emission color and mask. */
    struct PrincipledVolumeParams {
      int     enabled = 0;
      float   emissionStrength = 0.f;

      struct DD {
        int     enabled;
        float   emissionStrength;
      };

      inline DD getDD(Device *device) const
      {
        DD dd;
        dd.enabled = enabled;
        dd.emissionStrength = emissionStrength;
        (void)device;
        return dd;
      }
    };

    /*! Radiance added at a real collision. The emission transfer function
        independently controls emission color and mask. */
    inline __rtc_device
    vec3f principledCollisionEmission(float scalar,
                                      const TransferFunction::DD &emissionXf,
                                      int emissionXfActive,
                                      const PrincipledVolumeParams::DD &p)
    {
      if (p.emissionStrength <= 0.f || !emissionXfActive)
        return vec3f(0.f);

      const vec4f e = emissionXf.map(scalar);
      if (e.w <= 0.f)
        return vec3f(0.f);
      return p.emissionStrength * e.w * getPos(e);
    }

  }
}
