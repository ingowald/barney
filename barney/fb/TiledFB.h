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

#define FB_NO_PEER_ACCESS 1

namespace barney {

  struct FrameBuffer;
  
#if DENOISE_NORMAL
  /*! for now, do a 48 bit half3 representation; shoul eventually go
      to 32 or 16, but lets at least try how much this really helps in
      the denoiser */
  struct CompressedNormal {
    inline __device__ void set(vec3f v) { x = v.x; y = v.y; z = v.z; }
    inline __device__ vec3f get() const { return vec3f(x,y,z); }
    inline __device__ float4 get4f() const { return make_float4(x,y,z,0.f); }
    inline __device__ float3 get3f() const { return make_float3(x,y,z); }
    half x,y,z;
  };
#endif
  
  void float4ToBGBA8(uint32_t  *finalFB,
# if DENOISE_OIDN  
                     float3    *inputBeforeDenoising,
                     float     *alphas,
                     float3    *float3s,
# else
                     float4    *inputBeforeDenoising,
                     float4    *float4s,
# endif
                     float      denoisedWeight,
                     vec2i      numPixels);
  
  enum { tileSize = 32 };
  enum { pixelsPerTile = tileSize*tileSize };

  struct AccumTile {
    float4 accum[pixelsPerTile];
    float  depth[pixelsPerTile];
#if DENOISE_NORMAL
    CompressedNormal normal[pixelsPerTile];
#endif
  };
  struct FinalTile {
    uint32_t rgba[pixelsPerTile];
#if DENOISE_NORMAL
    half     scale[pixelsPerTile];
    CompressedNormal normal[pixelsPerTile];
#endif
    float    depth[pixelsPerTile];
  };
  struct TileDesc {
    vec2i lower;
  };

  struct TiledFB {
    typedef std::shared_ptr<TiledFB> SP;
    static SP create(Device::SP device, FrameBuffer *owner);

    TiledFB(Device::SP device, FrameBuffer *owner);
    virtual ~TiledFB();

    void resize(vec2i newSize);
    void free();

    /*! write this tiledFB's tiles into given "final" frame buffer
        (i.e., a plain 2D array of numPixels.x*numPixels.y RGBA8
        pixels) */
    static
    void writeFinalPixels(
#if DENOISE
# if DENOISE_OIDN
                          float3    *finalFB,
                          float     *finalAlpha,
# else
                          float4    *finalFB,
# endif
#else
                          uint32_t  *finalFB,
#endif
                          float     *finalDepth,
# if DENOISE_NORMAL
#  if DENOISE_OIDN
                          float3    *finalNormal,
#  else
                          float4    *finalNormal,
#  endif
# endif
                          vec2i      numPixels,
                          FinalTile *finalTiles,
                          TileDesc  *tileDescs,
                          int        numTiles,
                          bool       showCrosshairs);
    
    void finalizeTiles();

    /*! number of (valid) pixels */
    vec2i numPixels       = { 0,0 };

    /*! number of tiles to cover the entire frame buffer; some on the
      right/bottom may be partly filled */
    vec2i numTiles        = { 0, 0 };
    int   numActiveTiles  = 0;
    /*! lower-left pixel coordinate for given tile ... */
    TileDesc  *tileDescs  = 0;
    AccumTile *accumTiles = 0;
    FinalTile *finalTiles = 0;
    FrameBuffer *const owner;
    Device::SP  const device;
  };

}
