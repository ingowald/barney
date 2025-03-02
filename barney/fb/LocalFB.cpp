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
  {
  }

  void LocalFB::resize(vec2i size,
                       uint32_t channels)
  {
    Device *frontDev = getDenoiserDevice();
    auto rtc = frontDev->rtc;
    FrameBuffer::resize(size,channels);
    
    if (gatheredTilesOnOwner.compressedTiles)
      rtc->freeMem(gatheredTilesOnOwner.compressedTiles);
    if (gatheredTilesOnOwner.tileDescs)
      rtc->freeMem(gatheredTilesOnOwner.tileDescs);
  
    // do NOT set active device - it's whatever the app used!
    // SetActiveDevice forDuration(perDev[0]->device);
    int sumTiles = 0;
    for (auto device : *devices)
      sumTiles += getFor(device)->numActiveTiles;

    gatheredTilesOnOwner.numActiveTiles = sumTiles;
    gatheredTilesOnOwner.compressedTiles
      = (CompressedTile *)rtc->allocMem(sumTiles*sizeof(CompressedTile));
    gatheredTilesOnOwner.tileDescs
      = (TileDesc *)rtc->allocMem(sumTiles*sizeof(TileDesc));
    sumTiles = 0;
    for (auto device : *devices) {
      auto devFB = getFor(device);
      device->rtc->copyAsync(gatheredTilesOnOwner.tileDescs+sumTiles,
                             devFB->tileDescs,
                             devFB->numActiveTiles*sizeof(TileDesc));
      sumTiles += devFB->numActiveTiles;
    }
    for (auto device : *devices)
      device->sync();
  }
  
  void LocalFB::ownerGatherCompressedTiles()
  {
    // do NOT set active device - it's whatever the app used!
    // SetActiveDevice forDuration(perDev[0]->device);
    int sumTiles = 0;
    for (auto device : *devices) {
      auto devFB = getFor(device);
      device->rtc->copyAsync(gatheredTilesOnOwner.compressedTiles+sumTiles,
                             devFB->compressedTiles,
                             devFB->numActiveTiles*sizeof(CompressedTile));
      sumTiles += devFB->numActiveTiles;
    }
    gatheredTilesOnOwner.numActiveTiles = sumTiles;
    
    for (auto device : *devices)
      device->sync();
  }
  
  LocalFB::~LocalFB()
  {
    auto frontDev = getDenoiserDevice();
    frontDev->rtc->freeMem(gatheredTilesOnOwner.compressedTiles);
    frontDev->rtc->freeMem(gatheredTilesOnOwner.tileDescs);
  }
  
}
