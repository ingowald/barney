// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

#pragma once

#include "mori/DeviceGroup.h"

namespace mori {

  struct Material {
    vec3f diffuseColor;
  };
  
  struct Geom {
    typedef std::vector<Geom> SP;

    Geom(DeviceContext *device,
         const Material &material)
      : device(device),
        material(material)
    {}

    Material       material;
    OWLGeom        owlGeom = 0;
    DeviceContext *device  = 0;
  };
  
}
