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
// #ifdef BARNEY_BACKEND_OPTIX
// #include <optix.h>
// #include <optix_stubs.h>
// #endif

namespace barney {

  struct FrameBuffer;

#if 0
  struct Denoiser {
    typedef std::shared_ptr<Denoiser> SP;
    static SP create(FrameBuffer *fb);
    
    Denoiser(FrameBuffer *fb) : fb(fb) {};
    virtual ~Denoiser() {};
    virtual void resize() = 0;
    virtual void run() = 0;
    FrameBuffer *const fb;
  };
#endif
  
  struct FrameBuffer : public SlottedObject {

    FrameBuffer(Context *context,
                const DevGroup::SP &devices,
                const bool isOwner);
    virtual ~FrameBuffer();

    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "<FrameBuffer(base)>"; }

    bool set1i(const std::string &member, const int &value) override;

    virtual void resize(vec2i size, uint32_t channels);
    virtual void resetAccumulation() {  /* whatever we may have in compressed tiles is dirty */ accumID = 0; }
    void freeResources();

    void finalizeTiles();
    void finalizeFrame();
    virtual void ownerGatherCompressedTiles() = 0;

    void read(BNFrameBufferChannel channel,
              void *hostPtr,
              BNDataType requestedFormat);

    struct {
      CompressedTile *compressedTiles     = 0;
      TileDesc       *tileDescs      = 0;
      int             numActiveTiles = 0;
    } gatheredTilesOnOwner;

    TiledFB *getFor(Device *device);
    struct PLD {
      TiledFB::SP tiledFB;
    };
    PLD *getPLD(Device *device);
    
    std::vector<PLD> perLogical;

    void *getPointer(BNFrameBufferChannel channel);
    
    /*! on owner, take the 'gatheredTilesOnOwner', and unpack them into
        linear color, depth, alpha, and normal channels, so denoiser
        can then run on it */
    void unpackTiles();
    
    /*! if true, then we have already gathered, re-arranged, possibly
        denoised, tone mapped, etcpp whatever tiles the various
        devices may have accumulated so far, and we can simply copy
        pixels to the app */
    bool dirty = false;
    /*! compressed color buffer, in array-(not tiled) order, after
        denoising - only on owner. All denoiser implementations will
        generate exactly this format, so the compressed bnFrameBufferRead()
        can then just copy from this format */

    vec4f *denoisedColor = 0;
    
    vec4f *linearColor = 0;
    float *linearDepth = 0;
    vec3f *linearNormal = 0;
    
    vec2i numPixels = {-1,-1};

    Device *getDenoiserDevice() const;
    rtc::Denoiser *denoiser;
    // Denoiser::SP denoiser;

    uint32_t    accumID = 0;
    const bool  isOwner;
    bool  showCrosshairs = false;
  };
}
