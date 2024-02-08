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
#include "barney/fb/FrameBuffer.h"
#include "barney/Model.h"

namespace barney {
  
  Context::~Context()
  {
    hostOwnedHandles.clear();
    std::map<Object::SP,int> hostOwnedHandles;

    perDG.clear();

    owlContextDestroy(globalContextAcrossAllGPUs);
  }
  
  void Context::releaseHostReference(Object::SP object)
  {
    auto it = hostOwnedHandles.find(object);
    if (it == hostOwnedHandles.end())
      throw std::runtime_error
        ("trying to bnRelease() a handle that either does not "
         "exist, or that the app (no lnoger) has any valid references on");

    const int remainingReferences = --it->second;

    if (remainingReferences == 0) {
      // remove the std::shared-ptr handle:
      it->second = {};
      // and make barney forget that it ever had this object 
      hostOwnedHandles.erase(it);
    }
  }
  
  void Context::addHostReference(Object::SP object)
  {
    auto it = hostOwnedHandles.find(object);
    if (it == hostOwnedHandles.end())
      throw std::runtime_error
        ("trying to bnAddReference() to a handle that either does not "
         "exist, or that the app (no lnoger) has any valid primary references on");
    
    // add one ref count:
    it->second++;
  }
  
  void Context::generateRays(const barney::Camera &camera,
                             FrameBuffer *fb)
  {
    assert(fb);
    int accumID=fb->accumID;

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

  void Context::shadeRaysLocally(FrameBuffer *fb, int generation)
  {
    BARNEY_CUDA_SYNC_CHECK();
    for (int localID=0; localID<devices.size(); localID++) {
      auto dev = devices[localID];
      dev->shadeRays_launch(fb->perDev[localID].get(),generation);
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
    // ------------------------------------------------------------------
    // launch all in parallel ...
    // ------------------------------------------------------------------
    for (int localID=0; localID<devices.size(); localID++) {
      auto dev = devices[localID];
      dev->traceRays_launch(model);
    }
    // ------------------------------------------------------------------
    // ... and sync 'til all are done
    // ------------------------------------------------------------------
    for (auto dev : devices) dev->sync();
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
                            FrameBuffer *fb,
                            int pathsPerPixel)
  {
    if (!isActiveWorker)
      return;

    for (auto &pd : perDG) 
      pd.devGroup->update();

    // iw - todo: add wave-front-merging here.
    for (int p=0;p<pathsPerPixel;p++) {
      generateRays(camera,fb);
      for (auto dev : devices) dev->launch_sync();

      for (int generation=0;true;generation++) {
        traceRaysGlobally(model);
        for (auto dev : devices) dev->launch_sync();

        shadeRaysLocally(fb, generation);
        for (auto dev : devices) dev->launch_sync();
      
        const int numActiveGlobally = numRaysActiveGlobally();
        if (numActiveGlobally > 0)
          continue;
    
        break;
      }
      ++ fb->accumID;
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

    globalContextAcrossAllGPUs
      = owlContextCreate((int32_t*)gpuIDs.data(),(int)gpuIDs.size());

    if (!isActiveWorker) 
      // not an active worker: no device groups etc, just create a
      // single default device
      return;


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
    int gpusPerDG = (int)gpuIDs.size() / numDGs;
    std::vector<std::vector<int>> gpuInDG(numDGs);
    perDG.resize(numDGs);
    for (int ldgID=0;ldgID<numDGs;ldgID++) {
      auto &dg = perDG[ldgID];
      dg.dataGroupID = dataGroupIDs[ldgID];
      for (int j=0;j<gpusPerDG;j++)
        dg.gpuIDs.push_back(gpuIDs[ldgID*gpusPerDG+j]);
      dg.devGroup = std::make_shared
        <DevGroup>(ldgID,
                   dg.gpuIDs,
                   globalIndex*numDGs+ldgID,
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

