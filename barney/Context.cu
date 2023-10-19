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
#include "barney/DeviceGroup.h"
#include "barney/FrameBuffer.h"
#include "barney/Model.h"

namespace barney {
  
  void Context::generateRays(const barney::Camera &camera,
                             FrameBuffer *fb)
  {
    assert(fb);
    int accumID=0;
    // ------------------------------------------------------------------
    // launch all GPUs to do their stuff
    // ------------------------------------------------------------------
    for (int localID=0; localID<devices.size(); localID++) {
      barney::TiledFB *mfb = fb->perDev[localID].get();
      assert(mfb);
      
      auto dev = devices[localID];
      assert(dev);
      
      dev->rays.resetWriteQueue();
      dev->generateRays_launch(mfb,camera,accumID);
    }
    // ------------------------------------------------------------------
    // wait for all GPUs' completion
    // ------------------------------------------------------------------
    for (int localID=0; localID<devices.size(); localID++) {
      auto dev = devices[localID];
      dev->generateRays_sync();
      // std::cout << "num rays generated in dev " << localID << " is " << dev->rays.numActive << std::endl;
    }
  }
  
  /*! returns how many rays are active in all ray queues, across all
    devices and, where applicable, across all ranks */
  int Context::numRaysActiveLocally()
  {
    int numActive = 0;
    for (auto dev : devices)
      numActive += dev->rays.numActiveRays();
    return numActive;
  }

  void Context::shadeRaysLocally(FrameBuffer *fb)
  {
    BARNEY_CUDA_SYNC_CHECK();
    for (int localID=0; localID<devices.size(); localID++) {
      auto dev = devices[localID];
      dev->shadeRays_launch(fb->perDev[localID].get());
    }
    BARNEY_CUDA_SYNC_CHECK();
    for (int localID=0; localID<devices.size(); localID++) {
      auto dev = devices[localID];
      dev->shadeRays_sync();
    }
    BARNEY_CUDA_SYNC_CHECK();
  }
  
  void Context::traceRaysLocally(Model *model)
  {
    for (int localID=0; localID<devices.size(); localID++) {
      auto dev = devices[localID];
      dev->traceRays_launch(model);
    }
    for (int localID=0; localID<devices.size(); localID++) {
      auto dev = devices[localID];
      dev->launch_sync();
    }
    BARNEY_CUDA_SYNC_CHECK();
    // for (int localID=0; localID<devices.size(); localID++) {
    //   auto dev = devices[localID];
    //   dev->rays.numActive = 0;
    // }
  }

  void Context::traceRaysGlobally(Model *model)
  {
    while (true) {
      traceRaysLocally(model);
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
    for (int localID = 0; localID < devices.size(); localID++)
      // (will set active GPU internally)
      fb->perDev[localID]->finalizeTiles();

    for (int localID = 0; localID < devices.size(); localID++)
      devices[localID]->launch_sync();
  }
  
  void Context::renderTiles(Model *model,
                            const Camera &camera,
                            FrameBuffer *fb)
  {
    if (!isActiveWorker)
      return;

    for (auto &pd : perDG) 
      pd.devGroup->update();
    
    generateRays(camera,fb);
    while (true) {
      traceRaysGlobally(model);
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
    : isActiveWorker(!dataGroupIDs.empty())
  {
    if (gpuIDs.empty())
      throw std::runtime_error("error - no GPUs...");
    
    if (!isActiveWorker) {
      // // not an active worker: no device groups etc, just create a
      // // single default device
      // Device::SP dev
      //   = std::make_shared
      //   <barney::Device>(nullptr,gpuIDs[0],-1,
      //                    globalIndex,globalIndexStep);
      // devices.push_back(dev);
      return;
    }


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
    std::vector<std::vector<int>> gpuInDG(numDGs);
    perDG.resize(numDGs);
    for (int dgID=0;dgID<numDGs;dgID++) {
      auto &dg = perDG[dgID];
      for (int j=0;j<gpusPerDG;j++)
        dg.gpuIDs.push_back(gpuIDs[dgID*gpusPerDG+j]);
      dg.devGroup = std::make_shared
        <DevGroup>(dgID,
                   dg.gpuIDs,
                   globalIndex*numDGs+dgID,
                   globalIndexStep*numDGs);
      for (auto dev : dg.devGroup->devices)
        devices.push_back(std::make_shared<DeviceContext>(dev));
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

    for (int localID = 0; localID < devices.size(); localID++) {
      auto dev = devices[localID];
      dev->rays.ensureRayQueuesLargeEnoughFor(fb->perDev[localID].get());
    }
  }
  
}

