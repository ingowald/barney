// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

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

