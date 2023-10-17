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
#include "mori/TiledFB.h"

namespace barney {

  using mori::TileDesc;

  struct FrameBuffer : public Object {

    FrameBuffer(Context *context, const bool isOwner);
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "<FrameBuffer(base)>"; }
    
    virtual void resize(vec2i size, uint32_t *hostFB);
    
    std::vector<mori::TiledFB::SP> moris;
    
    vec2i       numPixels   = { 0,0 };
    uint32_t   *finalFB     = 0;
    uint32_t   *hostFB      = 0;
    Context    *const context;
    const bool  isOwner;
  };
}
