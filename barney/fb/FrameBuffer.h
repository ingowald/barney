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

namespace BARNEY_NS {

  struct FrameBuffer;

  struct FrameBuffer : barney_api::FrameBuffer {

    FrameBuffer(Context *context,
                const DevGroup::SP &devices,
                const bool isOwner);
    virtual ~FrameBuffer();

    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "<FrameBuffer(base)>"; }

    /*! resize frame buffer to given number of pixels and the
        indicated types of channels; color will only ever get queries
        in 'colorFormat'. Channels is a bitmask compoosed of
        or'ed-together BN_FB_xyz channel flags; only those bits that
        are set may get queried by the application (ie those that are
        not set do not have to be stored or even computed */
    void resize(BNDataType colorFormat,
                vec2i size,
                uint32_t channels) override;
    void resetAccumulation() override
    {  /* whatever we may have in compressed tiles is dirty */ accumID = 0; }
    void freeResources();

    bool needHitIDs() const;
    
    void finalizeTiles();
    void finalizeFrame();

    /*! gather color (and optionally, if not null) linear normal, from
        all GPUs (and ranks). lienarColor and lienarNormal are
        device-writeable 2D linear arrays of numPixel size;
        linearcolor may be null. */
    virtual void gatherColorChannel(/*float4 or rgba8*/void *linearColor,
                                    BNDataType gatherType,
                                    // can be null:
                                    vec3f *linearNormal) = 0;
    
    /*! read one of the auxiliary (not color or normal) buffers into
      the given app memory; this will at the least incur some
      reformatting from tiles to linear (if local node), possibly
      some gpu-gpu transfer (local node w/ more than one gpu) and
      possibly some mpi communication (distFB) */
    virtual void gatherAuxChannel(BNFrameBufferChannel channel) = 0;
    virtual void writeAuxChannel(void *stagingArea,
                                 BNFrameBufferChannel channel) = 0;
    
    /*! read given frame buffer channel into given application memory
        (which may be either host or device memory), in requested
        format. Requeseted color format (currently) has to match the
        color channel format specified during resize() */
    void read(BNFrameBufferChannel channel,
              void *appDataPtr,
              BNDataType requestedFormat) override;

    // struct {
    //   /*! _all_ tile descriptors across all GPUs - either all GPUs in
    //     single node (if run non-mpi) or across all nodes */
    //   TileDesc *tileDescs       = 0;
    //   int       sumTiles = 0;
    // } onOwner;
    

    TiledFB *getFor(Device *device);
    struct PLD {
      TiledFB::SP tiledFB;
    };
    PLD *getPLD(Device *device);
    
    std::vector<PLD> perLogical;

    /*! staging area for gathering/writing pixels into, will be Nx*Ny
        pixels of type as provided specified during resize. This may
        point to either float4 or rgba8 depending on requested
        format */
    void *linearColorChannel = 0;
    
    /*! staging area for re-assembling any other channel; note this
        may store either float or uint32 data */
    void *linearAuxChannel = 0;
    
    /*! the channels we're supposed to have (as asked for on the latest resize()) */
    uint32_t   channels = 0;
    BNDataType colorChannelFormat = BN_DATA_UNDEFINED;
    vec2i      numPixels = {-1,-1};

    Device *getDenoiserDevice() const;

    /*! gather color (and normal, if required for denoising),
      (re-)format into a linear buffer, perform denoising (if
      required), convert to requested format, and copy to the
      application pointed provided */
    void readColorChannel(void *appMemory,
                          BNDataType requestedFormat);

    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool set1i(const std::string &member, const int &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    
    /*! points to the rtc denoiser object we've created. Can be null
        if denoising is disabled in cmake, or if the given rtc backend
        doesn't support denoising (eg, old optix version, oidn not
        found during compiling, etc). Also see \see
        enableDenoising. */
    rtc::Denoiser *denoiser = 0;

    /*! whether to "in principle" do denoising. Denoising will still
     require an rtc backend that does have a denoiser (\see denoiser
     field), but this allows a user to disable denoising at runtime */
    bool enableDenoising = 1;

    /*! how many samples per pixels have already been accumulated in
        this frame buffer's accumulation buffer. Note this is counted
        in *samples*, not *frames*. */
    uint32_t    accumID = 0;

    /*! kernel that converts from a linear device-side float4 format
        to a linear device-side ufixed8 format */
    rtc::ComputeKernel2D *linear_toFixed8 = 0;
    
    const bool  isOwner;
    bool  showCrosshairs = false;
    DevGroup::SP const devices;
  };
}
