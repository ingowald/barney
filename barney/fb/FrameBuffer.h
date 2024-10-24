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

  struct FrameBuffer : public Object {

    FrameBuffer(Context *context, const bool isOwner);
    virtual ~FrameBuffer();

    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "<FrameBuffer(base)>"; }

    bool set1i(const std::string &member, const int &value);

    virtual void resize(vec2i size,
                        uint32_t *hostFB,
                        float    *hostDepth);
    virtual void resetAccumulation() { accumID = 0; }
    void freeResources();

#if FB_NO_PEER_ACCESS
    virtual void ownerGatherFinalTiles() = 0;
#endif

    std::vector<TiledFB::SP> perDev;

    vec2i       numPixels   = { 0,0 };
    // the final frame buffer RGBA8 that we can definitely write into
    // - might be our own staged copy if we can't write into host
    // supplied one
    uint32_t   *finalFB     = 0;
    uint32_t   *hostFB      = 0;

    // depth buffer: same as for color buffer we have two differnt
    // poitners here - one that we can defintiely use in device code
    // (finalDepth), and one that the app wants to eventually have the
    // values in (might be on host). these _can_ be the same, but may
    // not be
    float      *finalDepth  = 0;
    float      *hostDepth   = 0;


#if DENOISE
    bool                 denoiserCreated = false;
    OptixDenoiser        denoiser = {};
    float4              *denoiserInput   = 0;
    float4              *denoiserOutput  = 0;
#if DENOISE_NORMAL
    float4              *denoiserNormal  = 0;
#endif
    OptixDenoiserOptions denoiserOptions;
    void                *denoiserScratch = 0;
    void                *denoiserState   = 0;
    OptixDenoiserSizes denoiserSizes;
    void denoise();
#endif
    
    uint32_t    accumID     = 0;
    const bool  isOwner;
          bool  showCrosshairs;
  };
}
