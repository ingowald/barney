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

#include "QuadLight.h"

namespace barney {
  
  QuadLight::DD QuadLight::getDD(const affine3f &instanceXfm) const
  {
    DD dd;
    dd.corner = xfmPoint(instanceXfm,staged.corner);
    dd.edge0 = xfmVector(instanceXfm,staged.edge0);
    dd.edge1 = xfmVector(instanceXfm,staged.edge1);
    dd.emission = staged.emission;
    dd.area = length(cross(dd.edge0,dd.edge1));
    return dd;
  }
  
  bool QuadLight::set3f(const std::string &member, const vec3f &value) 
  {
    if (member == "corner") {
      staged.corner = value;
      return true;
    }
    if (member == "edge0") {
      staged.edge0 = value;
      return true;
    }
    if (member == "edge1") {
      staged.edge1 = value;
      return true;
    }
    if (member == "emission") {
      staged.emission = value;
      return true;
    }
    return false;
  }

}
