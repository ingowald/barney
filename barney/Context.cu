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
    for (int localID=0; localID<fb->perGPU.size(); localID++) {
      mori::TiledFB *mfb = fb->perGPU[localID];
      assert(mfb);
      
      auto dev = perGPU[localID];
      assert(dev);
      
      dev->rays.resetWriteQueue();
      dev->generateRays_launch(mfb,camera,accumID);
    }
    // ------------------------------------------------------------------
    // wait for all GPUs' completion
    // ------------------------------------------------------------------
    for (int localID=0; localID<fb->perGPU.size(); localID++) {
      auto dev = perGPU[localID];
      dev->generateRays_sync();
    }
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
                   const std::vector<int> &gpuIDs)
    : dataGroupIDs(dataGroupIDs),
      gpuIDs(gpuIDs),
      isActiveWorker(!dataGroupIDs.empty())
  {
    if (gpuIDs.empty())
      throw std::runtime_error("error - no GPUs...");
    perGPU.resize(gpuIDs.size());
    for (int localID=0;localID<gpuIDs.size();localID++) {
      DeviceContext *devCon = new DeviceContext(gpuIDs[localID]);
      devCon->tileIndexScale  = gpuIDs.size();
      devCon->tileIndexOffset = localID;
      devCon->gpuID = gpuIDs[localID];
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

