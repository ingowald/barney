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

#include "barney/Texture.h"
#include "barney/Context.h"
#include "barney/DataGroup.h"

namespace barney {

  Texture::Texture(DataGroup *owner,
                   BNTexelFormat texelFormat,
                   vec2i size,
                   const void *texels,
                   BNTextureFilterMode  filterMode,
                   BNTextureAddressMode addressMode,
                   BNTextureColorSpace  colorSpace)
    : Object(owner->context)
  {
    assert(OWL_TEXEL_FORMAT_RGBA8   == (int)BN_TEXEL_FORMAT_RGBA8);
    assert(OWL_TEXEL_FORMAT_RGBA32F == (int)BN_TEXEL_FORMAT_RGBA32F);
    
    assert(OWL_TEXTURE_NEAREST == (int)BN_TEXTURE_NEAREST);
    assert(OWL_TEXTURE_LINEAR  == (int)BN_TEXTURE_LINEAR);
    
    assert(OWL_TEXTURE_WRAP   == (int)BN_TEXTURE_WRAP);
    assert(OWL_TEXTURE_CLAMP  == (int)BN_TEXTURE_CLAMP);
    assert(OWL_TEXTURE_BORDER == (int)BN_TEXTURE_BORDER);
    assert(OWL_TEXTURE_MIRROR == (int)BN_TEXTURE_MIRROR);
    
    owlTex = owlTexture2DCreate(owner->getOWL(),
                                (OWLTexelFormat)texelFormat,
                                size.x,size.y,
                                texels,
                                (OWLTextureFilterMode)filterMode,
                                (OWLTextureAddressMode)addressMode,
                                // (OWLTextureColorSpace)colorSpace
  OWL_COLOR_SPACE_LINEAR
                                );
  }

}
