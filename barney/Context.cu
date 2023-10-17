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
    assert(fb);
    int accumID=0;
    // ------------------------------------------------------------------
    // launch all GPUs to do their stuff
    // ------------------------------------------------------------------
    for (int localID=0; localID<moris.size(); localID++) {
      mori::TiledFB *mfb = fb->moris[localID].get();
      assert(mfb);
      
      auto mori = moris[localID];
      assert(mori);
      
      mori->rays.resetWriteQueue();
      mori->generateRays_launch(mfb,camera,accumID);
    }
    // ------------------------------------------------------------------
    // wait for all GPUs' completion
    // ------------------------------------------------------------------
    for (int localID=0; localID<moris.size(); localID++) {
      auto mori = moris[localID];
      mori->generateRays_sync();
    }
  }
  
  /*! returns how many rays are active in all ray queues, across all
    devices and, where applicable, across all ranks */
  int Context::numRaysActiveLocally()
  {
    int numActive = 0;
    for (auto mori : moris)
      numActive += mori->rays.numActiveRays();
    return numActive;
  }

  void Context::shadeRaysLocally(FrameBuffer *fb)
  {
    for (int localID=0; localID<moris.size(); localID++) {
      auto mori = moris[localID];
      mori->shadeRays_launch(fb->moris[localID].get());
    }
    for (int localID=0; localID<moris.size(); localID++) {
      auto mori = moris[localID];
      mori->launch_sync();
    }
  }
  
  void Context::traceRaysLocally()
  {
    for (int localID=0; localID<moris.size(); localID++) {
      auto mori = moris[localID];
      mori->rays.numActive = 0;
    }
  }

  void Context::traceRaysGlobally()
  {
    while (true) {
      traceRaysLocally();
      const bool needMoreTracing = forwardRays();
      if (needMoreTracing)
        continue;
      break;
    }
  }

  void Context::finalizeTiles(FrameBuffer *fb)
  {
    // ------------------------------------------------------------------
    // tell each device to finalize its rendered accum tiles
    // ------------------------------------------------------------------
    for (int localID = 0; localID < moris.size(); localID++)
      // (will set active GPU internally)
      fb->moris[localID]->finalizeTiles();

    for (int localID = 0; localID < moris.size(); localID++)
      moris[localID]->launch_sync();
  }
  
  void Context::renderTiles(Model *model,
                            const mori::Camera &camera,
                            FrameBuffer *fb)
  {
    if (!isActiveWorker)
      return;
    
    generateRays(camera,fb);
    while (true) {
      traceRaysGlobally();
      shadeRaysLocally(fb);
      if (numRaysActiveGlobally() > 0)
        continue;
      break;
    }
  }

  
  Context::Context(const std::vector<int> &dataGroupIDs,
                   const std::vector<int> &gpuIDs,
                   int globalIndex,
                   int globalIndexStep)
    : // dataGroupIDs(dataGroupIDs),
      // gpuIDs(gpuIDs),
      isActiveWorker(!dataGroupIDs.empty())
  {
    PING;
    if (gpuIDs.empty())
      throw std::runtime_error("error - no GPUs...");
    moris.resize(gpuIDs.size());
    PING;
    PRINT(this);
    PRINT(gpuIDs.size());
    PRINT(gpuIDs.data());
    for (int localID=0;localID<gpuIDs.size();localID++) {
      PRINT(localID);
      PRINT(gpuIDs[localID]);
      mori::Device::SP moriDev
        = std::make_shared
        <mori::Device>(gpuIDs[localID],
                       globalIndex * gpuIDs.size() + localID,
                       globalIndexStep * gpuIDs.size());
      PING; fflush(0);
      mori::MoriContext::SP
        mori = std::make_shared
        <mori::MoriContext>(moriDev);
      moris[localID] = mori;
    }

    PING;
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
      int numDGs = dataGroupIDs.size();
      int gpusPerDG = gpuIDs.size() / numDGs;
      perDG.resize(numDGs);
      for (int localID=0;localID<numDGs;localID++) {
        std::vector<mori::Device::SP> devs;
        perDG[localID].dataGroupID = dataGroupIDs[localID];
        for (int j=0;j<gpusPerDG;j++)
          devs.push_back(moris[localID*gpusPerDG+j]->device);
        perDG[localID].devGroup
          = mori::DevGroup::create(devs);
      }
    }
    PING;
  }

  Model *Context::createModel()
  {
    return initReference(Model::create(this));
  }

  void Context::ensureRayQueuesLargeEnoughFor(FrameBuffer *fb)
  {
    if (!isActiveWorker)
      return;

    for (int localID = 0; localID < moris.size(); localID++) {
      auto mori = moris[localID];
      mori->rays.ensureRayQueuesLargeEnoughFor(fb->moris[localID].get());
    }
  }
  
}

