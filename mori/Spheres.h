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

#include "mori/Geometry.h"

namespace mori {

  /* the host side representation */
  struct Spheres : public Geom {
    Spheres(DeviceContext *device,
            const Material &material,
            const vec3f *origins,
            int numOrigins,
            const float *radii,
            float defaultRadius);

    struct OnDev {
      vec3f   *origins;
      float   *radii;
      float    defaultRadius;
      Material material;
    };
    
    static OWLGeomType createGeomType(DeviceContext *device);

    OWLBuffer   originsBuffer = 0;
    OWLBuffer   radiiBuffer   = 0;
    float       defaultRadius = .1f;
  };
  
}
