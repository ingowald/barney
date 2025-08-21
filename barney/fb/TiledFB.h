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

#include "barney/DeviceGroup.h"
#include "barney/common/half.h"
#include "barney/render/HitIDs.h"
#include "barney/Context.h"

namespace BARNEY_NS {
  
  struct FrameBuffer;
  
  struct AuxChannelTile {
    union { uint32_t ui[pixelsPerTile]; float f[pixelsPerTile]; };
  };
  
  
  struct AccumTile {
    vec4f  accum[pixelsPerTile];
    // float  depth[pixelsPerTile];
    vec3f  normal[pixelsPerTile];
    // int    primID[pixelsPerTile];
    // int    objID[pixelsPerTile];
    // int    instID[pixelsPerTile];
  };

  struct AuxTiles {
    AuxChannelTile *depth  = 0;
    AuxChannelTile *primID = 0;
    AuxChannelTile *instID = 0;
    AuxChannelTile *objID  = 0;
  };

  /*! describes the lower-left corner of each logical tile */
  struct TileDesc {
    vec2i lower;
  };
  
  struct TiledFB {
    typedef std::shared_ptr<TiledFB> SP;
    static SP create(Device *device,
                     /*! device for the gpu that the app
                       lives on, if different from current
                       GPU, and if no peer access is
                       available */
                     Device *appDevice,
                     FrameBuffer *owner);

    TiledFB(Device *device,
            /*! device on which the app runs and will do map(). if
                null, this will be ignored, and the tile finalization
                operations will just read from local device and write
                to final frame buffer (assuming peer access to this
                memory). If non null, the tiledFB will first copy
                tiles to appDevice, and then run tile conversion
                there. */
            Device *appDevice,
            FrameBuffer *owner);
    virtual ~TiledFB();

    void resize(uint32_t channels,
                vec2i newSize);
    void free();

    /*! take this GPU's tiles, and write those tiles' color (and
        optionally normal) channels into the linear frame buffers
        provided. The linearColor is guaranteed to be non-null, and to
        be numPixels.x*numPixels.y vec4fs; linearNormal may be
        null. Linear buffers may live on another GPU, but are
        guaranteed to be on the same node. */
    void linearizeColorAndNormal(void  *linearColor,
                                 BNDataType format,
                                 vec3f *linearNormal,
                                 float  accumScale);

    /*! linearize given array's aux tiles, on given device. this can be
      used either for local GPUs on a single node, or on the owner
      after it reveived all worker tiles */
    static void linearizeAuxTiles(Device *device,
                                  void *linearOut,
                                  vec2i numPixels,
                                  AuxChannelTile *tilesIn,
                                  TileDesc       *descsIn,
                                  int numTiles);

    /*! linearize _this gpu's_ channels */
    void linearizeAuxChannel(void *linearChannel,
                             BNFrameBufferChannel whichChannel);
    
    /*! number of (valid) pixels */
    vec2i numPixels       = { 0,0 };
    
    /*! number of tiles to cover the entire frame buffer; some on the
      right/bottom may be partly filled, and this particlar GPUs instance
      will only own some of those tiles */
    vec2i numTiles        = { 0, 0 };
    
    /*! number of tiles that ***this GPU*** owns */
    int   numActiveTilesThisGPU  = 0;
    
    /*! lower-left pixel coordinate for given tile ... */
    TileDesc          *tileDescs     = 0;
    TileDesc          *appTileDescs  = 0;
    AccumTile         *accumTiles    = 0;
    AccumTile         *appAccumTiles = 0;
    AuxTiles           auxTiles;
    AuxTiles           appAuxTiles;

    
    FrameBuffer *const owner;
    Device      *const device;
    
    /*! device on which the app runs and will do map(). if null, this
      will be ignored, and the tile finalization operations will just
      read from local device and write to final frame buffer (assuming
      peer access to this memory). If non null, the tiledFB will first
      copy tiles to appDevice, and then run tile conversion there. */
    Device      *const appDevice;
  };

}
