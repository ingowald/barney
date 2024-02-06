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

#include "barney/Texture.h"

namespace barney {
  struct Material {
    
    struct DD {
      vec3f baseColor;
      float ior;
      float transmission;
      cudaTextureObject_t colorTexture;
      cudaTextureObject_t alphaTexture;
    };
    vec3f baseColor;
    float transmission;
    float ior;
    Texture::SP colorTexture;
    Texture::SP alphaTexture;
    
    static void addVars(std::vector<OWLVarDecl> &vars, int base);
    void set(OWLGeom geom) const;
  };

}
