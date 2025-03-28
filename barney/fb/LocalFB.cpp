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
    Device *frontDev = getDenoiserDevice();
    auto rtc = frontDev->rtc;
    FrameBuffer::resize(colorFormat,size,channels);
    
    if (onOwner.tileDescs)
      rtc->freeMem(onOwner.tileDescs);

    onOwner.sumTiles = 0;
    for (auto device : *devices)
      onOwner.sumTiles += getFor(device)->numActiveTiles;

    onOwner.tileDescs
      = (TileDesc *)rtc->allocMem(onOwner.sumTiles*sizeof(TileDesc));
    TileDesc *dst = onOwner.tileDescs;
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      auto devFB = getFor(device);
      device->rtc->copyAsync(dst,
                             devFB->tileDescs,
                             devFB->numActiveTiles*sizeof(TileDesc));
      dst += devFB->numActiveTiles;
    }
    
    for (auto device : *devices)
      device->sync();
  }
  
  LocalFB::~LocalFB()
  {
    auto frontDev = getDenoiserDevice();
    SetActiveGPU forDuration(frontDev);
    // frontDev->rtc->freeMem(.compressedTiles);
    frontDev->rtc->freeMem(onOwner.tileDescs);
  }

  /*! gather color (and optionally, if not null) linear normal, from
    all GPUs (and ranks). lienarColor and lienarNormal are
    device-writeable 2D linear arrays of numPixel size;
    linearcolor may be null. */
  void LocalFB::gatherColorChannel(/*float4 or rgba8*/void *linearColor,
                                   BNDataType gatherType,
                                   vec3f *linearNormal)
  {
    float accumScale = 1.f/accumID;
    for (auto device : *devices) {
      getFor(device)->linearizeColorAndNormal
        (linearColor,gatherType,linearNormal,accumScale);
    }
    for (auto device : *devices)
      device->rtc->sync();
  }

  /*! read one of the auxiliary (not color or normal) buffers into
    the given app memory; this will at the least incur some
    reformatting from tiles to linear (if local node), possibly
    some gpu-gpu transfer (local node w/ more than one gpu) and
    possibly some mpi communication (distFB) */
  void LocalFB::gatherAuxChannel(void *stagingArea,
                                 BNFrameBufferChannel whichChannel) 
  {
    for (auto device : *devices)
      getFor(device)->linearizeAuxChannel(stagingArea,whichChannel);
  }
  
} // ::BARNEY_NS
