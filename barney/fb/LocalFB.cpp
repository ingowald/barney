// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/fb/LocalFB.h"

namespace BARNEY_NS {
  
  LocalFB::LocalFB(Context *context,
                   const DevGroup::SP &devices)
    : FrameBuffer(context, devices, true)
  {}

  void LocalFB::resize(BNDataType colorFormat,
                       vec2i size,
                       uint32_t channels)
  {
    PING;
    Device *frontDev = getDenoiserDevice();
    for (auto device : *devices) {
      auto devFB = getFor(device);
    }
    
    PING;
    auto rtc = frontDev->rtc;
    FrameBuffer::resize(colorFormat,size,channels);
    
    if (onOwner.tileDescs)
      rtc->freeMem(onOwner.tileDescs);
    PING;

    onOwner.sumTiles = 0;
    for (auto device : *devices)
      onOwner.sumTiles += getFor(device)->numActiveTilesThisGPU;

    PING;
    onOwner.tileDescs
      = (TileDesc *)rtc->allocMem(onOwner.sumTiles*sizeof(TileDesc));
    TileDesc *dst = onOwner.tileDescs;
    PING;
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      auto devFB = getFor(device);
      device->rtc->copyAsync(dst,
                             devFB->tileDescs,
                             devFB->numActiveTilesThisGPU*sizeof(TileDesc));
      dst += devFB->numActiveTilesThisGPU;
    }
    
    PING;
    for (auto device : *devices)
      device->sync();
    PING;
  }
  
  LocalFB::~LocalFB()
  {
    auto frontDev = getDenoiserDevice();
    SetActiveGPU forDuration(frontDev);
    frontDev->rtc->freeMem(onOwner.tileDescs);
  }

  /*! gather color (and optionally, if not null) linear normal, from
    all GPUs (and ranks). lienarColor and linearNormal are
    device-writeable 2D linear arrays of numPixel size;
    linearcolor may be null. */
  void LocalFB::gatherColorChannel(/*float4 or rgba8*/void *linearColor,
                                   BNDataType gatherType,
                                   vec3f *linearNormal)
  {
    float accumScale = 1.f/accumID;
    for (auto device : *devices) {
      auto tfb = getFor(device);
      tfb->linearizeColorAndNormal
        (linearColor,gatherType,linearNormal,accumScale);
    }
  }

  /*! read one of the auxiliary (not color or normal) buffers into
    the given app memory; this will at the least incur some
    reformatting from tiles to linear (if local node), possibly
    some gpu-gpu transfer (local node w/ more than one gpu) and
    possibly some mpi communication (distFB) */
  void LocalFB::gatherAuxChannel(BNFrameBufferChannel whichChannel)
  {
    /* nothing to do , we can always write from tiledFBs */
  }
  
  void LocalFB::writeAuxChannel(void *stagingArea,
                                BNFrameBufferChannel whichChannel) 
  {
    for (auto device : *devices)
      getFor(device)->linearizeAuxChannel(stagingArea,whichChannel);
    for (auto device : *devices)
      device->sync();
  }
  
} // ::BARNEY_NS
