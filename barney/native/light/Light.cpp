// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/light/Light.h"
#include "native/ModelSlot.h"
#include "native/Context.h"

#include "native/light/QuadLight.h"
#include "native/light/DirLight.h"
#include "native/light/EnvMap.h"

namespace BARNEY_NS {
  namespace native {
    
    Light::Light(Device *device)
      : Object(device)
    {}
  
    Light::SP Light::create(Device *device,
                            const std::string &type)
    {
      if (type == "directional")
        return std::make_shared<DirLight>(device);
      if (type == "quad")
        return std::make_shared<QuadLight>(device);
      if (type == "point")
        return std::make_shared<PointLight>(device);
      if (type == "envmap")
        return std::make_shared<EnvMapLight>(device);
    
      return {};
    }

    // ==================================================================
    bool Light::set3f(const std::string &member, const vec3f &value)
    {
      if (member == "color") {
        color = value;
        return true;
      }
      return false;
    }
  
    // ==================================================================
  
    // ==================================================================

  }
}
