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

#include "barney/Context.h"
#include "mori/DeviceGroup.h"
#include "barney/FrameBuffer.h"
#include "barney/Model.h"

namespace barney {
  
  // __global__ void render_testFrame(vec2i fbSize,
  //                                  mori::AccumTile *tiles,
  //                                  vec2i numTiles,
  //                                  int tileIndexOffset,
  //                                  int tileIndexScale)
  // {
  //   int localTileIdx = blockIdx.x;
  //   int globalTileIdx = tileIndexScale * localTileIdx + tileIndexOffset;
  //   int tile_y = globalTileIdx / numTiles.x;
  //   int tile_x = globalTileIdx - tile_y * numTiles.x;
  //   int ix = threadIdx.x + tile_x * mori::tileSize;
  //   int iy = threadIdx.y + tile_y * mori::tileSize;

  //   bool dbg = (ix == 118 && iy == 123);
  //   if (dbg)
  //     printf("(%i %i) fb %i %i tile %i %i\n",
  //            ix,iy,fbSize.x,fbSize.y,tile_x,tile_y);
    
  //   if (ix >= fbSize.x) return;
  //   if (iy >= fbSize.y) return;
  //   mori::AccumTile &tile = tiles[localTileIdx];

  //   // int sx = 13*17;
  //   // int sy = 11*19;
  //   // float r = (ix % sx)/(sx-1.f);
  //   // float g = (iy % sy)/(sy-1.f);
  //   float r = ix / (fbSize.x-1.f);
  //   float g = iy / (fbSize.y-1.f);
  //   float b = 1.f - (ix+iy)/(fbSize.x+fbSize.y-1.f);

  //   tile.accum[threadIdx.y*mori::tileSize+threadIdx.x] = make_float4(r,g,b,1.f);

  //   // appFB[ix + iy * fbSize.x] = owl::make_rgba(vec3f(r,g,b));
  // }
  

  Context::Context(const std::vector<int> &dataGroupIDs,
                   const std::vector<int> &gpuIDs)
    : dataGroupIDs(dataGroupIDs),
      gpuIDs(gpuIDs)
  {
    deviceContexts.resize(gpuIDs.size());
    for (int localID=0;localID<gpuIDs.size();localID++) {
      DeviceContext *devCon = new DeviceContext;
      devCon->tileIndexScale  = gpuIDs.size();
      devCon->tileIndexOffset = localID;
      devCon->gpuID = gpuIDs[localID];
      devCon->owl   = owlContextCreate(&devCon->gpuID,1);
      devCon->stream = owlContextGetStream(devCon->owl,0);
      deviceContexts[localID] = devCon;
    }

    if (gpuIDs.size() < dataGroupIDs.size())
      throw std::runtime_error("not enough GPUs ("
                               +std::to_string(gpuIDs.size())
                               +") for requested num data groups ("
                               +std::to_string(gpuIDs.size())
                               +")");
    if (gpuIDs.size() % dataGroupIDs.size())
      throw std::runtime_error("requested num GPUs is not a multiple of "
                               "requested num data groups");
    int numMoris = dataGroupIDs.size();
    int gpusPerMori = gpuIDs.size() / numMoris;
    moris.resize(numMoris);
    for (int moriID=0;moriID<numMoris;moriID++) {
      std::vector<int> gpusThisMori(gpusPerMori);
      for (int j=0;j<gpusPerMori;j++)
        gpusThisMori[j] = gpuIDs[moriID*gpusPerMori+j];
      moris[moriID] = mori::DeviceGroup::create(gpusThisMori);
    }
  }

  Model *Context::createModel()
  {
    return initReference(Model::create(this));
  }
      
}

