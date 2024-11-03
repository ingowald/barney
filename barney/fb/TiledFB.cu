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

#include "barney/fb/TiledFB.h"
#include "barney/fb/FrameBuffer.h"
#include <optix.h>
#include <optix_function_table.h>
#include <optix_stubs.h>

// #include "optix_host.h"
// #include "optix_stubs.h"

namespace barney {

  TiledFB::SP TiledFB::create(Device::SP device, FrameBuffer *owner)
  {
    return std::make_shared<TiledFB>(device, owner);
  }

  TiledFB::TiledFB(Device::SP device, FrameBuffer *owner)
    : device(device),
      owner(owner)
  {}

  TiledFB::~TiledFB()
  { free(); }

  void TiledFB::free()
  {
    SetActiveGPU forDuration(device);
    if (accumTiles)  {
      BARNEY_CUDA_CALL(Free(accumTiles));
      accumTiles = nullptr;
    }
    if (finalTiles) {
      BARNEY_CUDA_CALL(Free(finalTiles));
      finalTiles = nullptr;
    }
    if (tileDescs) {
      BARNEY_CUDA_CALL(Free(tileDescs));
      tileDescs = nullptr;
    }
  }

  __global__ void setTileCoords(TileDesc *tileDescs,
                                int numActiveTiles,
                                vec2i numTiles,
                                int globalIndex,
                                int globalIndexStep)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= numActiveTiles)
      return;

    int tileID = tid * globalIndexStep + globalIndex;

    int tile_x = tileID % numTiles.x;
    int tile_y = tileID / numTiles.x;
    tileDescs[tid].lower = vec2i(tile_x*tileSize,tile_y*tileSize);
  }

  void TiledFB::resize(vec2i newSize)
  {
    free();
    SetActiveGPU forDuration(device);

    numPixels = newSize;
    numTiles  = divRoundUp(numPixels,vec2i(tileSize));
    numActiveTiles
      = device
      ? divRoundUp(numTiles.x*numTiles.y - device->globalIndex,
                   device->globalIndexStep)
      : 0;
#if 0
    BARNEY_CUDA_CALL(MallocManaged(&accumTiles, numActiveTiles * sizeof(AccumTile)));
    BARNEY_CUDA_CALL(MallocManaged(&finalTiles, numActiveTiles * sizeof(FinalTile)));
    BARNEY_CUDA_CALL(MallocManaged(&tileDescs,  numActiveTiles * sizeof(TileDesc)));
#else
    BARNEY_CUDA_CALL(Malloc(&accumTiles, numActiveTiles * sizeof(AccumTile)));
    BARNEY_CUDA_CALL(Malloc(&finalTiles, numActiveTiles * sizeof(FinalTile)));
    BARNEY_CUDA_CALL(Malloc(&tileDescs,  numActiveTiles * sizeof(TileDesc)));
#endif

    BARNEY_CUDA_SYNC_CHECK();
    if (numActiveTiles)
      setTileCoords<<<divRoundUp(numActiveTiles,1024),1024,0,
      device?device->launchStream:0>>>
        (tileDescs,numActiveTiles,numTiles,
         device->globalIndex,device->globalIndexStep);
    BARNEY_CUDA_SYNC_CHECK();
  }

  // ==================================================================

  __global__ void g_finalizeTiles(FinalTile *finalTiles,
                                  AccumTile *accumTiles,
                                  float      accumScale)
  {
    int pixelID = threadIdx.x;
    int tileID  = blockIdx.x;

    vec4f color = vec4f(accumTiles[tileID].accum[pixelID])*accumScale;
#if DENOISE
    float scale = reduce_max(color);
    color *= 1./scale;
    finalTiles[tileID].scale[pixelID] = scale;
    finalTiles[tileID].normal[pixelID].set(accumTiles[tileID].normal[pixelID]);
#else
    color.x = sqrtf(color.x);
    color.y = sqrtf(color.y);
    color.z = sqrtf(color.z);
#endif

    uint32_t rgba32
      = owl::make_rgba(color);

    finalTiles[tileID].rgba[pixelID] = rgba32;
    finalTiles[tileID].depth[pixelID] = accumTiles[tileID].depth[pixelID];
  }

  /*! write this tiledFB's tiles into given "final" frame buffer
    (i.e., a plain 2D array of numPixels.x*numPixels.y RGBA8
    pixels) */
  void TiledFB::finalizeTiles()
  {
    SetActiveGPU forDuration(device);
    if (numActiveTiles > 0)
      g_finalizeTiles<<<numActiveTiles,pixelsPerTile,0,device->launchStream>>>
        (finalTiles,accumTiles,1.f/(owner->accumID));
  }


  // ==================================================================
// #if DENOISE
//   __global__ void g_float4ToBGBA8(uint32_t  *finalFB,
// # if DENOISE_OIDN  
//                                   float3    *inputBeforeDenoising,
//                                   float     *alphas,
//                                   float3    *float3s,
// # else
//                                   float4    *inputBeforeDenoising,
//                                   float4    *float4s,
// #endif
//                                   float      denoisedWeight,
//                                   vec2i      numPixels)
//   {
//     int ix = threadIdx.x+blockIdx.x*blockDim.x;
//     int iy = threadIdx.y+blockIdx.y*blockDim.y;
//     if (ix >= numPixels.x) return;
//     if (iy >= numPixels.y) return;
    
//     int pid = ix+numPixels.x*iy;
// # if DENOISE_OIDN
//     float4 v;
//     (float3&)v = float3s[pid];
//     v.w = alphas[pid];
// #else
//     float4 v = float4s[pid];
// #  if 1
//     float4 v2 = inputBeforeDenoising[pid];
//     v
//       = denoisedWeight*(const vec4f&)v
//       + (1.f-denoisedWeight)*(const vec4f&)v2;
// #  endif
// # endif
//     v.x = sqrtf(v.x);
//     v.y = sqrtf(v.y);
//     v.z = sqrtf(v.z);
//     finalFB[pid] = make_rgba(vec4f(v));
//   }
  
//   void float4ToBGBA8(uint32_t  *finalFB,
// #if DENOISE_OIDN
//                      float3    *inputBeforeDenoising,
//                      float     *alphas,
//                      float3    *float3s,
// #else
//                      float4    *inputBeforeDenoising,
//                      float4    *float4s,
// #endif
//                      float      denoisedWeight,
//                      vec2i      numPixels)  
//   {
//     vec2i tileSize = 32;
//     g_float4ToBGBA8
//       <<<divRoundUp(numPixels,tileSize),tileSize>>>
//       (finalFB,
// # if DENOISE_OIDN
//        inputBeforeDenoising,alphas,float3s,
// # else
//        inputBeforeDenoising,float4s,
// # endif
//        denoisedWeight,
//        numPixels);
//   }    
//   __global__ void g_writeFinalPixels(
// #if DENOISE_OIDN
//                                      float4    *finalFB,
//                                      float     *finalAlpha,
// #else
//                                      float4    *finalFB,
// #endif
//                                      float     *finalDepth,
// #if DENOISE
// # if DENOISE_OIDN
//                                      float3    *finalNormal,
// # else
//                                      float4    *finalNormal,
// # endif
// #endif
//                                      vec2i      numPixels,
//                                      FinalTile *finalTiles,
//                                      TileDesc  *tileDescs,
//                                      bool       showCrosshairs)
//   {
//     int tileID = blockIdx.x;
//     int ix = threadIdx.x + tileDescs[tileID].lower.x;
//     int iy = threadIdx.y + tileDescs[tileID].lower.y;
//     if (ix < 0 || ix >= numPixels.x) return;
//     if (iy < 0 || iy >= numPixels.y) return;

//     uint32_t pixelValue
//       = finalTiles[tileID].rgba[threadIdx.x + tileSize*threadIdx.y];
//     float scale
//       = finalTiles[tileID].scale[threadIdx.x + tileSize*threadIdx.y];
//     pixelValue |= 0xff000000;

//     uint32_t ofs = ix + numPixels.x*iy;

//     bool isCenter_x = ix == numPixels.x/2;
//     bool isCenter_y = iy == numPixels.y/2;
//     bool isCrossHair = (isCenter_x || isCenter_y) && !(isCenter_x && isCenter_y);

//     float a = ((pixelValue >> 24) & 0xff) / 255.f;
//     float b = ((pixelValue >> 16) & 0xff) / 255.f;
//     float g = ((pixelValue >>  8) & 0xff) / 255.f;
//     float r = ((pixelValue >>  0) & 0xff) / 255.f;
//     // if (ix == 128 && iy == 128)
//     //   printf("%f %f %f %f\n",r,g,b,a);
// # if DENOISE_OIDN
//     finalFB[ofs]
//       = showCrosshairs && isCrossHair
//       ? make_float3(1.f,0.f,0.f)
//       : make_float3(scale*r,scale*g,scale*b);
//     finalAlpha[ofs] = a;
// # else
//     finalFB[ofs]
//       = showCrosshairs && isCrossHair
//       ? make_float4(1.f,0.f,0.f,1.f)
//       : make_float4(scale*r,scale*g,scale*b,a);
// #endif

//     if (finalDepth)
//       finalDepth[ofs] = finalTiles[tileID].depth[threadIdx.x + tileSize*threadIdx.y];
// #  if DENOISE_OIDN
//     finalNormal[ofs]
//       = finalTiles[tileID].normal[threadIdx.x + tileSize*threadIdx.y].get3f();
// #  else
//     finalNormal[ofs]
//       = finalTiles[tileID].normal[threadIdx.x + tileSize*threadIdx.y].get4f();
// #  endif
//   }
// #else
//   __global__ void g_writeFinalPixels(uint32_t  *finalFB,
//                                      float     *finalDepth,
//                                      vec2i      numPixels,
//                                      FinalTile *finalTiles,
//                                      TileDesc  *tileDescs,
//                                      bool       showCrosshairs)
//   {
//     int tileID = blockIdx.x;
//     int ix = threadIdx.x + tileDescs[tileID].lower.x;
//     int iy = threadIdx.y + tileDescs[tileID].lower.y;
//     if (ix < 0 || ix >= numPixels.x) return;
//     if (iy < 0 || iy >= numPixels.y) return;

//     uint32_t pixelValue
//       = finalTiles[tileID].rgba[threadIdx.x + tileSize*threadIdx.y];
//     pixelValue |= 0xff000000;

    
//     uint32_t ofs = ix + numPixels.x*iy;

//     bool isCenter_x = ix == numPixels.x/2;
//     bool isCenter_y = iy == numPixels.y/2;
//     bool isCrossHair = (isCenter_x || isCenter_y) && !(isCenter_x && isCenter_y);

//     finalFB[ofs]
//       = showCrosshairs && isCrossHair
//       ? 0xff0000ff
//       : pixelValue;

//     if (finalDepth)
//       finalDepth[ofs] = finalTiles[tileID].depth[threadIdx.x + tileSize*threadIdx.y];
//   }
// #endif
  
//   void TiledFB::writeFinalPixels(
// #if DENOISE
// # if DENOISE_OIDN
//                                  float3    *finalFB,
//                                  float     *finalAlpha,
// # else
//                                  float4    *finalFB,
// # endif
// #else
//                                  uint32_t  *finalFB,
// #endif
//                                  float     *finalDepth,
// #if DENOISE
// #  if DENOISE_OIDN
//                                  float3    *finalNormal,
// #  else
//                                  float4    *finalNormal,
// #  endif
// #endif
//                                  vec2i      numPixels,
//                                  FinalTile *finalTiles,
//                                  TileDesc  *tileDescs,
//                                  int        numTiles,
//                                  bool       showCrosshairs)
//   {
//     if (finalFB == 0) throw std::runtime_error("invalid finalfb of null!");

   
//     /*! do NOT set active GPU: app might run on a different GPU than
//         what we think of as GPU 0, and taht may or may not be
//         writeable by what we might think of a "first" gpu */
//     // SetActiveGPU forDuration(device);

//     if (numTiles > 0)
//       g_writeFinalPixels
//         <<<numTiles,vec2i(tileSize)>>>
//       //   ,0,
//       // device?device->launchStream:0>>>
//         (finalFB,
// #if DENOISE_OIDN
//          finalAlpha,
// #endif
//          finalDepth,
// #if DENOISE
//          finalNormal,
// #endif
//          numPixels,
//          finalTiles,tileDescs, showCrosshairs);
//   }
  
}
