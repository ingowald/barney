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

#pragma once

#include "barney/Object.h"
#include "barney/common/Data.h"
#include "barney/common/Texture.h"

namespace barney {

  struct ModelSlot;
  
  struct Light : public SlottedObject {
    typedef std::shared_ptr<Light> SP;

    /*! what we return, during rendering, when we sample a light
        source */
    struct Sample {
      /* direction _to_ light */
      vec3f direction;
      /*! radiance coming _from_ dir */
      vec3f radiance;
      /*! distance to this light sample */
      float distance;
      /*! pdf of sample that was chosen */
      float pdf = 0.f;
    };
  
    
    Light(ModelSlot *owner) : SlottedObject(owner) {}

    std::string toString() const override { return "Light<>"; }
    
    static Light::SP create(ModelSlot *owner, const std::string &name);
  };

};
