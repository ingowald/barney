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
  
  void Context::generateRays(const mori::Camera &camera,
                             FrameBuffer *fb)
  {
    PING;
    assert(fb);
    int accumID=0;
    for (int localID=0; localID<fb->perGPU.size(); localID++) {
      mori::TiledFB *mfb = fb->perGPU[localID];
      assert(mfb);
      
      auto dev = perGPU[localID];
      assert(dev);
      
      dev->rays.resetWriteQueue();
      dev->generateRays_launch(mfb,camera,accumID);
    }
    PING;
    for (int localID=0; localID<fb->perGPU.size(); localID++) {
      auto dev = perGPU[localID];
      dev->generateRays_sync();
    }
    PING;
  }
  
  /*! returns how many rays are active in all ray queues, across all
    devices and, where applicable, across all ranks */
  int Context::numRaysActiveLocally()
  {
    int numActive = 0;
    for (auto dev : perGPU)
      numActive += dev->rays.numActiveRays();
    return numActive;
  }

  void Context::shadeRaysLocally(FrameBuffer *fb)
  {
    for (int localID=0; localID<perGPU.size(); localID++) {
      auto dev = perGPU[localID];
      dev->shadeRays_launch(fb->perGPU[localID]);
    }
    for (int localID=0; localID<perGPU.size(); localID++) {
      auto dev = perGPU[localID];
      dev->sync();
    }
  }
  
  void Context::traceRaysLocally()
  {
    for (int localID=0; localID<perGPU.size(); localID++) {
      auto dev = perGPU[localID];
      dev->rays.numActive = 0;
    }
  }

  void Context::traceRaysGlobally()
  {
    while (true) {
      traceRaysLocally();
      
      PING;
      bool needMoreTracing = forwardRays();
      PING;
      if (!needMoreTracing)
        break;
      PING;
    }
  }

  void Context::finalizeTiles(FrameBuffer *fb)
  {
    // ------------------------------------------------------------------
    // tell each device to finalize its rendered accum tiles
    // ------------------------------------------------------------------
    for (int localID = 0; localID < gpuIDs.size(); localID++)
      // (will set active GPU internally)
      fb->perGPU[localID]->finalizeTiles();

    for (int localID = 0; localID < gpuIDs.size(); localID++)
      perGPU[localID]->sync();
  }
  
  void Context::renderTiles(Model *model,
                            const mori::Camera &camera,
                            FrameBuffer *fb)
  {
    if (!isActiveWorker)
      return;

    std::cout<< "==================================================================" << std::endl;
    PING; PRINT(&camera); PRINT(fb);
    
    PING;
    generateRays(camera,fb);
    while (true) {
      PING;
      traceRaysGlobally();

      PING;
      shadeRaysLocally(fb);
      
      PING;
      int numActive = numRaysActiveGlobally();
      PRINT(numActive);
      if (numActive == 0)
        break;
      
      PING;
    }
    PING;
    std::cout<< "==================================================================" << std::endl;
  }


  // __global__ void g_renderTiles_rayDir(mori::AccumTile *tiles,
  //                                      mori::TileDesc  *tileDescs,
  //                                      int numTiles,
  //                                      vec2i fbSize,
  //                                      mori::Camera camera)
  // {
  //   int tileID = blockIdx.x;
  //   vec2i tileOffset = tileDescs[tileID].lower;
  //   int ix = threadIdx.x + tileOffset.x;
  //   int iy = threadIdx.y + tileOffset.y;

  //   if (ix >= fbSize.x) return;
  //   if (iy >= fbSize.y) return;
  //   mori::AccumTile &tile = tiles[tileID];

  //   vec3f dir
  //     = (const vec3f&)camera.dir_00
  //     + (ix+.5f)*(const vec3f&)camera.dir_du
  //     + (iy+.5f)*(const vec3f&)camera.dir_dv;
    
  //   vec3f color = abs(normalize(dir));
  //   tile.accum[threadIdx.y*mori::tileSize+threadIdx.x]
  //     = make_float4(color.x,color.y,color.z,1.f);
  // }
  
  // void renderTiles_rayDir(Context *context,
  //                            int localID,
  //                            Model *model,
  //                            FrameBuffer *fb,
  //                            const BNCamera *camera)
  // {
  //   auto &devFB = *fb->perGPU[localID];
  //   auto device = devFB.device;
    
  //   SetActiveGPU forDuration(device->gpuID);
  //   g_renderTiles_rayDir
  //     <<<devFB.numActiveTiles,vec2i(mori::tileSize),0,device->stream>>>
  //     (devFB.accumTiles,
  //      devFB.tileDescs,
  //      devFB.numActiveTiles,
  //      devFB.numPixels,
  //      *camera);
  // }
  
  // void renderTiles(Context *context,
  //                  int localID,
  //                  Model *model,
  //                  FrameBuffer *fb,
  //                  const BNCamera *camera)
  // {
  //   renderTiles_rayDir(context,localID,model,fb,camera);
  // }    


  // __global__ void g_renderTiles_testFrame(mori::AccumTile *tiles,
  //                                         mori::TileDesc  *tileDescs,
  //                                         int numTiles,
  //                                         vec2i fbSize)
  // {
  //   int tileID = blockIdx.x;
  //   vec2i tileOffset = tileDescs[tileID].lower;
  //   int ix = threadIdx.x + tileOffset.x;
  //   int iy = threadIdx.y + tileOffset.y;
    
    
  //   if (ix >= fbSize.x) return;
  //   if (iy >= fbSize.y) return;
  //   mori::AccumTile &tile = tiles[tileID];

  //   float r = ix / (fbSize.x-1.f);
  //   float g = iy / (fbSize.y-1.f);
  //   float b = 1.f - (ix+iy)/(fbSize.x+fbSize.y-1.f);

  //   tile.accum[threadIdx.y*mori::tileSize+threadIdx.x] = make_float4(r,g,b,1.f);
  // }
  
  // void renderTiles_testFrame(Context *context,
  //                            int localID,
  //                            Model *model,
  //                            FrameBuffer *fb,
  //                            const BNCamera *camera)
  // {
  //   auto &devFB = *fb->perGPU[localID];
  //   auto device = devFB.device;
    
  //   SetActiveGPU forDuration(device->gpuID);
  //   g_renderTiles_testFrame
  //     <<<devFB.numActiveTiles,vec2i(mori::tileSize),0,device->stream>>>
  //     (devFB.accumTiles,
  //      devFB.tileDescs,
  //      devFB.numActiveTiles,
  //      devFB.numPixels);
  // }
  
  Context::Context(const std::vector<int> &dataGroupIDs,
                   const std::vector<int> &gpuIDs)
    : dataGroupIDs(dataGroupIDs),
      gpuIDs(gpuIDs),
      isActiveWorker(!dataGroupIDs.empty())
  {
    if (gpuIDs.empty())
      throw std::runtime_error("error - no GPUs...");
    perGPU.resize(gpuIDs.size());
    for (int localID=0;localID<gpuIDs.size();localID++) {
      DeviceContext *devCon = new DeviceContext;
      devCon->tileIndexScale  = gpuIDs.size();
      devCon->tileIndexOffset = localID;
      devCon->gpuID = gpuIDs[localID];
      devCon->owl   = owlContextCreate(&devCon->gpuID,1);
      devCon->stream = owlContextGetStream(devCon->owl,0);
      perGPU[localID] = devCon;
    }

    if (isActiveWorker) {
      if (gpuIDs.size() < dataGroupIDs.size())
        throw std::runtime_error("not enough GPUs ("
                                 +std::to_string(gpuIDs.size())
                                 +") for requested num data groups ("
                                 +std::to_string(dataGroupIDs.size())
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
  }

  Model *Context::createModel()
  {
    return initReference(Model::create(this));
  }

  void Context::ensureRayQueuesLargeEnoughFor(FrameBuffer *fb)
  {
    if (!isActiveWorker)
      return;

    for (int localID = 0; localID < perGPU.size(); localID++) {
      auto dev = perGPU[localID];
      dev->rays.ensureRayQueuesLargeEnoughFor(fb->perGPU[localID]);
    }
  }
  
}

