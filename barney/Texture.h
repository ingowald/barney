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
#include <barney.h>

namespace barney {

  struct DataGroup;

  /*! geometry in the form of a regular triangle mesh - vertex
      positoins array, vertex indices array, verex normals, and
      texcoords */
  struct Texture : public Object {
    typedef std::shared_ptr<Texture> SP;

    // struct DD {
    //   cudaTextureObject_t tex;
    // };
    
    Texture(DataGroup *owner,
            BNTexelFormat texelFormat,
            vec2i size,
            const void *texels,
            BNTextureFilterMode  filterMode,
            BNTextureAddressMode addressMode,
            BNTextureColorSpace  colorSpace);

    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Texture{}"; }

    OWLTexture owlTex = 0;
  };

}
