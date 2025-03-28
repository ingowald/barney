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

#include "barney/fb/FrameBuffer.h"
#include "barney/common/MPIWrappers.h"

namespace BARNEY_NS {

  struct MPIContext;

  /*! for now, do a 48 bit half3 representation; shoul eventually go
      to 32 or 16, but lets at least try how much this really helps in
      the denoiser */
  struct CompressedNormal {
    inline __both__ vec4f get4f() const
    { vec3f v = get(); return vec4f(v.x,v.y,v.z,0.f); }
    inline __both__ vec3f get3f() const
    { vec3f v = get(); return vec3f(v.x,v.y,v.z); }
    inline __both__ void set(vec3f v) {
      if (v == vec3f(0.f)) { x = y = z = 0; return; }
      v = normalize(v);
      x = encode(v.x);
      y = encode(v.y);
      z = encode(v.z);
    }
    inline __both__ vec3f get() const { return vec3f(decode(x),decode(y),decode(z)); }
  private:
    inline __both__ int8_t encode(float f) const {
      f = clamp(f*128.f,-127.f,+127.f);
      return int8_t(f);
    }
    inline __both__ float decode(int8_t i) const {
      if (i==0) return 0.f;
      return (i<0) ? (i-.5f)*(1.f/128.f) : (i+.5f)*(1.f/128.f);
    }
    int8_t x,y,z;
  };
  
  struct CompressedColorTile {
    /*! rgb are ufixed8 and need to be multiplied by scale, a is
        ufixed8 w/o scale */
    uint32_t rgba[pixelsPerTile];
    half     scale[pixelsPerTile];
  };

  struct CompressedNormalTile {
    CompressedNormal normal[pixelsPerTile];
  }
  
  struct ExtraChannelTile {
    union { uint32_t ui; float f; } data[pixelsPerTile];
  };
  
  struct DistFB : public FrameBuffer {
    typedef std::shared_ptr<DistFB> SP;

    DistFB(MPIContext *context,
           const DevGroup::SP &devices,
           int owningRank);
    virtual ~DistFB() = default;
    
    void resize(vec2i size, uint32_t channels) override;

    void ownerGatherCompressedTiles() override;
    
    struct {
      CompressedTile   *compressedTiles = 0;
      ExtraChannelTile *extraChannelTiles = 0;
      TileDesc         *tileDescs       = 0;
      int               numActiveTiles  = 0;
    } gatheredTilesOnOwner;
    struct {
      std::vector<int> numTilesOnGPU;
      std::vector<int> firstTileOnGPU;
      int numGPUs;
    } ownerGather;
    // (world)rank that owns this frame buffer
    const int owningRank;
    const bool isOwner;
    const bool ownerIsWorker;
    CompressedTile  *compressedTiles = 0;
    MPIContext *context;
  };

}
