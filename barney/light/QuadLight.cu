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
    dd.corner = xfmPoint(instanceXfm,params.corner);
    dd.edge0 = xfmVector(instanceXfm,params.edge0);
    dd.edge1 = xfmVector(instanceXfm,params.edge1);
    dd.emission = params.emission;
    dd.area = length(cross(dd.edge0,dd.edge1));
    return dd;
  }
  
  bool QuadLight::set3f(const std::string &member, const vec3f &value) 
  {
    return false;
  }

}
