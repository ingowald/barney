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

  struct Texture : public Object {
    typedef std::shared_ptr<Texture> SP;

    Texture(DataGroup *owner,
            BNTexelFormat texelFormat,
            vec2i size,
            const void *texels,
            BNTextureFilterMode  filterMode,
            BNTextureAddressMode addressMode,
            BNTextureColorSpace  colorSpace);
    virtual ~Texture() = default;

    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Texture{}"; }

    OWLTexture owlTex = 0;
  };

  struct Texture3D : public DataGroupObject {
    typedef std::shared_ptr<Texture3D> SP;

    struct DD {
      cudaArray_t           voxelArray = 0;
      cudaTextureObject_t   texObj;
      cudaTextureObject_t   texObjNN;
    };
    /*! one tex3d per device */
    std::vector<DD> tex3Ds;
    
    Texture3D(DataGroup *owner,
              BNTexelFormat texelFormat,
              vec3i size,
              const void *texels,
              BNTextureFilterMode  filterMode,
              BNTextureAddressMode addressMode);
    virtual ~Texture3D() = default;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Texture3D{}"; }

    
  };

}
