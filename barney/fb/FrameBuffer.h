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
#include "barney/fb/TiledFB.h"
#include <optix.h>
#include <optix_stubs.h>

namespace barney {

  struct FrameBuffer;
  
  struct Denoiser {
    static std::shared_ptr<Denoiser> create(FrameBuffer *fb);
    
    Denoiser(FrameBuffer *fb);
    virtual ~Denoiser();
    virtual void resize() = 0;
    virtual void run() = 0;
    FrameBuffer *const fb;
  };
    
  struct FrameBuffer : public Object {

    FrameBuffer(Context *context, const bool isOwner);
    virtual ~FrameBuffer();

    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "<FrameBuffer(base)>"; }

    bool set1i(const std::string &member, const int &value);

    virtual void resize(vec2i size, uint32_t channels);
    virtual void resetAccumulation() {  /* whatever we may have in final tiles is dirty */ accumID = 0; }
    void freeResources();

    void finalizeFrame() { /* whatever we may have in final tiles is dirty */dirty = true; }
    virtual void ownerGatherFinalTiles() = 0;

    void read(BNFrameBufferChannel channel,
              void *hostPtr,
              BNDataType requestedFormat);
              
    std::vector<TiledFB::SP> perDev;

    /*! if true, then we have already gathered, re-arranged, possibly
        denoised, tone mapped, etcpp whatever tiles the various
        devices may have accumulated so far, and we can simply copy
        pixels to the app */
    bool dirty = false;
    /*! final color buffer, in array-(not tiled) order, after
        denoising - only on owner. All denoiser implementations will
        generate exactly this format, so the final bnFrameBufferRead()
        can then just copy from this format */
    float4 *finalColor = 0;
    /*! final depth buffer, in array-(not tiled) order, after
        denoising - only on owner. All denoiser implementations will
        generate exactly this format, so the final bnFrameBufferRead()
        can then just copy from this format*/
    float  *finalDepth = 0;
    vec2i numPixels = {-1,-1};

    std::shared_ptr<Denoiser> denoiser;

    uint32_t    accumID = 0;
    const bool  isOwner;
    bool  showCrosshairs = false;
  };
}
