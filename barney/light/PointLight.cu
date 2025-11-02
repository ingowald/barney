// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "PointLight.h"

namespace BARNEY_NS {
  
  PointLight::DD PointLight::getDD(const affine3f &instanceXfm) const
  {
    DD dd;
    dd.position  = xfmPoint(instanceXfm,position);
    dd.power     = power;
    dd.intensity = intensity;
    dd.color     = color;
    return dd;
  }

  bool PointLight::set1f(const std::string &member, const float &value) 
  {
    if (Light::set1f(member,value))
      return true;
    if (member == "power") {
      power = value;
      return true;
    }
    if (member == "intensity") {
      intensity = value;
      return true;
    }
    return false;
  }
  
  bool PointLight::set3f(const std::string &member, const vec3f &value) 
  {
    if (Light::set3f(member,value))
      return true;
    if (member == "position") {
      position = value;
      return true;
    }
    return false;
  }

  
}

