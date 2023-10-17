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

#include "barney/Context.h"
#include "mori/Spheres.h"

namespace barney {

  struct DataGroup;
  
  struct Spheres : public Object {
    typedef std::shared_ptr<Spheres> SP;

    Spheres(DataGroup *owner,
            const mori::Material &material,
            const vec3f *origins,
            int numOrigins,
            const float *radii,
            float defaultRadius);
    
    static SP create(DataGroup *owner,
                     const mori::Material &material,
                     const vec3f *origins,
                     int numOrigins,
                     const float *radii,
                     float defaultRadius)
    {
      return std::make_shared<Spheres>(owner,material,origins,numOrigins,
                                       radii,defaultRadius);
    }
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Spheres{}"; }
    
    DataGroup *const owner;
    mori::Spheres::SP mori;
  };

}
