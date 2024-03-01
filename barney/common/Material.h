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

#include "barney/Texture.h"

namespace barney {
  namespace materials {
    // ==================================================================
    struct Disney {
      // ---------------------------------
      struct InRayData {
        vec3h baseColor;
        half  ior;
        half  transmission;
      };
      // ---------------------------------
      struct DD {
        inline __device__ void make(InRayData &rayData, vec2f tc);
        
        vec3f baseColor;
        float ior;
        cudaTextureObject_t colorTexture;
        cudaTextureObject_t alphaTexture;
      };
    };
    // ==================================================================
    struct Matte {
      // ---------------------------------
      struct InRayData {
        vec3h reflectance;
      };
      // ---------------------------------
      struct DD {
        inline __device__ void make(InRayData &rayData, vec2f tc)
        { rayData.reflectance = reflectance; }
      };
      vec3f reflectance;
    };
  }

  namespace render {
    typedef enum { DISNEY, MATTE } MaterialType;

    struct Material {
      MaterialType type;
      union {
        Disney::InRayData disney;
        
    };
  }
  
  
  /*! a device-side material, as found in a shader binding table; this
      may still have device-specific 'bulk' data like textures */
  struct DeviceMaterial {
    MaterialType type;
    union {
      render::Matte matte;
      render::Disney
    };
  };
  
  /*! a (virtual) "barney material" as one interacts with throgh the
      API. this will eventually get specialized in derived classes,
      that then have to create device-materials to be put into actual
      SBTs of the geometries they are living in */
  struct Material : public SlottedObject {
  };

  struct DisneyMaterial : public Material {
    struct DD {
      vec3f baseColor;
      cudaTextureObject_t colorTexture;
      cudaTextureObject_t alphaTexture;
    };
    vec3f baseColor;
    Texture::SP colorTexture;
    Texture::SP alphaTexture;
    struct RayData {
    };
    struct DD {
      vec3h baseColor;
      half  ior;
      half  transmission;
    };
  };
  struct MatteMaterial : public Material{
    struct RayData {
      vec3f reflectance;
    };
    struct DD {
      vec3f reflectance;
    };
  };
  
}
