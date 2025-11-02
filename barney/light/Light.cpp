// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/light/Light.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"

#include "barney/light/QuadLight.h"
#include "barney/light/DirLight.h"
#include "barney/light/EnvMap.h"

namespace BARNEY_NS {

  Light::Light(Context *context,
               const DevGroup::SP &devices)
    : barney_api::Light(context),
      devices(devices)
  {}
  
  Light::SP Light::create(Context *context,
                          const DevGroup::SP &devices,
                          const std::string &type)
  {
    if (type == "directional")
      return std::make_shared<DirLight>(context,devices);
    if (type == "quad")
      return std::make_shared<QuadLight>(context,devices);
    if (type == "point")
      return std::make_shared<PointLight>(context,devices);
    if (type == "envmap")
      return std::make_shared<EnvMapLight>(context,devices);
    
    context->warn_unsupported_object("Light",type);
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
