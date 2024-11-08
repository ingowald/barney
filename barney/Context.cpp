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

#include "barney/Context.h"
#include "barney/DeviceGroup.h"
#include "barney/fb/FrameBuffer.h"
#include "barney/GlobalModel.h"
#include "barney/render/SamplerRegistry.h"
#include "barney/render/MaterialRegistry.h"

namespace barney {

  Context::~Context()
  {
    hostOwnedHandles.clear();
    std::map<Object::SP,int> hostOwnedHandles;

    perSlot.clear();

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
  
  void Context::generateRays(const barney::Camera::DD &camera,
                             Renderer *renderer,
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
      dev->generateRays_launch(mfb,camera,renderer->getDD(dev->device.get()),accumID);
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

  void Context::shadeRaysLocally(Renderer *renderer,
                                 GlobalModel *model,
                                 FrameBuffer *fb,
                                 int generation)
  {
    // BARNEY_CUDA_SYNC_CHECK();
    for (int localID=0; localID<devices.size(); localID++) {
      auto dev = devices[localID];
      dev->shadeRays_launch(renderer,model,fb->perDev[localID].get(),generation);
    }
    // BARNEY_CUDA_SYNC_CHECK();
    for (int localID=0; localID<devices.size(); localID++) {
      auto dev = devices[localID];
      dev->shadeRays_sync();
    }
    // BARNEY_CUDA_SYNC_CHECK();
  }
  
  void Context::traceRaysLocally(GlobalModel *model)
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

  void Context::traceRaysGlobally(GlobalModel *model)
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
  
  void Context::renderTiles(Renderer *renderer,
                            GlobalModel *model,
                            const Camera::DD &camera,
                            FrameBuffer *fb)
  {
    if (!isActiveWorker)
      return;

    for (auto &pd : perSlot) 
      pd.devGroup->update();

    // iw - todo: add wave-front-merging here.
    for (int p=0;p<renderer->pathsPerPixel;p++) {
#if 0
      std::cout << "====================== resetting accumid" << std::endl;
      fb->accumID = 0;
#endif
      double _t0 = getCurrentTime();
      generateRays(camera,renderer,fb);
      for (auto dev : devices) dev->launch_sync();

      for (int generation=0;true;generation++) {
        traceRaysGlobally(model);
        // do we need this here?
        for (auto dev : devices) dev->launch_sync();

        shadeRaysLocally(renderer, model, fb, generation);
        // no sync required here, shadeRays syncs itself.
        
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
    int numSlots = dataGroupIDs.size();
    int gpusPerSlot = (int)gpuIDs.size() / numSlots;
    std::vector<std::vector<int>> gpuInSlot(numSlots);
    perSlot.resize(numSlots);
    for (int lmsIdx=0;lmsIdx<numSlots;lmsIdx++) {
      std::vector<int> contextRanks;
      auto &dg = perSlot[lmsIdx];
      dg.modelRankInThisSlot = dataGroupIDs[lmsIdx];
      for (int j=0;j<gpusPerSlot;j++) {
        int localRank = lmsIdx*gpusPerSlot+j;
        contextRanks.push_back(localRank);
        dg.gpuIDs.push_back(gpuIDs[localRank]);
      }
      dg.devGroup = std::make_shared
        <DevGroup>(lmsIdx,
                   contextRanks,numSlots*gpusPerSlot,
                   dg.gpuIDs,
                   globalIndex*numSlots+lmsIdx,
                   globalIndexStep*numSlots);
      for (auto dev : dg.devGroup->devices) {
        devices.push_back(std::make_shared<DeviceContext>(dev));
        allDevices.push_back(dev);
      }
      
      dg.materialRegistry
        = std::make_shared<render::MaterialRegistry>(dg.devGroup);
      dg.samplerRegistry
        = std::make_shared<render::SamplerRegistry>(dg.devGroup);
    }
  }

  GlobalModel *Context::createModel()
  {
    return initReference(GlobalModel::create(this));
  }

  Renderer *Context::createRenderer()
  {
    return initReference(Renderer::create(this));
  }

  void Context::ensureRayQueuesLargeEnoughFor(FrameBuffer *fb)
  {
    if (!isActiveWorker)
      return;

    for (int localID = 0; localID < devices.size(); localID++) {
      auto dev = devices[localID];
      auto devFB = fb->perDev[localID].get();
      int upperBoundOnNumRays
        = 2 * (devFB->numActiveTiles+1) * barney::pixelsPerTile;
      dev->rays.reserve(upperBoundOnNumRays);
    }
  }

  int Context::contextSize() const
  {
    return devices.size();
  }
  
  
  std::shared_ptr<render::HostMaterial> Context::getDefaultMaterial(int slotID)
  {
    auto slot = getSlot(slotID);
    if (!slot->defaultMaterial)
      slot->defaultMaterial = std::make_shared<render::AnariMatte>(this,slotID);
    return slot->defaultMaterial;
  }

  const Context::PerSlot *Context::getSlot(int slot) const
  {
    if (slot < 0 || slot >= perSlot.size())
      throw std::runtime_error("tried to query an invalid slot!");
    return &perSlot[slot];
  }
  Context::PerSlot *Context::getSlot(int slot)
  {
    if (slot < 0 || slot >= perSlot.size())
      throw std::runtime_error("tried to query an invalid slot!");
    return &perSlot[slot];
  }

  OWLContext Context::getOWL(int slot) 
  {
    if (slot == -1) {
      PING;
      return globalContextAcrossAllGPUs;
    }
    return getDevGroup(slot)->owl;
  }

  const DevGroup *Context::getDevGroup(int slot) const
  {
    DevGroup *dg = getSlot(slot)->devGroup.get();
    assert(dg);
    return dg;
  }
  DevGroup *Context::getDevGroup(int slot) 
  {
    DevGroup *dg = getSlot(slot)->devGroup.get();
    assert(dg);
    return dg;
  }
  
  /*! helper function to print a warning when app tries to create anari
    object of certain kind and type that barney does not support */
  void Context::warn_unsupported_object(const std::string &kind,
                                        const std::string &type)
  {
    if (alreadyWarned.find(kind+"::"+type) != alreadyWarned.end())
      return;
    std::cout << OWL_TERMINAL_RED
              << "#bn: asked to create object of unknown/unsupported "
              <<  kind << " of type '" << type << "'"
              << " that I know nothing about"
              << OWL_TERMINAL_DEFAULT << std::endl;
    alreadyWarned.insert(kind+"::"+type);
  }
  
}

