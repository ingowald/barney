// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "DirLight.h"
#include "barney/common/math.h"

namespace BARNEY_NS {
  
  DirLight::DD DirLight::getDD(const affine3f &instanceXfm) const
  {
    DD dd;
    dd.direction = normalize(xfmVector(instanceXfm,direction));
    dd.color = color;
    dd.radiance
      = isnan(irradiance)
      ? radiance
      : irradiance;
    return dd;
  }

  bool DirLight::set1f(const std::string &member, const float &value) 
  {
    if (Light::set1f(member,value))
      return true;
    
    if (member == "irradiance") {
      irradiance = value;
      return true;
    }
    if (member == "radiance") {
      radiance = value;
      return true;
    }
    return false;
  }

  bool DirLight::set3f(const std::string &member, const vec3f &value) 
  {
    if (Light::set3f(member,value))
      return true;
    
    if (member == "direction") {
      direction = value;
      return true;
    }
    if (member == "radiance") {
      /* if - this is _not_ ANARI spec */
      std::cout << "#barney: WARNING - using float3 values for light (ir)radiance is deprecated" << std::endl;
      radiance = reduce_max(value);
      color = value/radiance;;
      return true;
    }
    if (member == "irradiance") {
      /* if - this is _not_ ANARI spec */
      std::cout << "#barney: WARNING - using float3 values for light (ir)radiance is deprecated" << std::endl;
      irradiance = reduce_max(value);
      color = value/irradiance;
      return true;
    }
    return false; 
  }
  
}

