// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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
#include "barney/render/RayQueue.h"
#include "barney/render/Sampler.h"
#include "barney/render/SamplerRegistry.h"
#include "barney/render/MaterialRegistry.h"
#include "barney/Camera.h"
#include "barney/render/Renderer.h"

namespace barney_api {
}  
// #endif
  
namespace BARNEY_NS {
  Context::Context(const std::vector<LocalSlot> &localSlots,
                   WorkerTopo::SP topo)
    : barney_api::Context(localSlots),
      isActiveWorker(!localSlots.empty() && localSlots[0].dataRank >= 0),
      topo(topo)
  {
    
    assert(!localSlots.empty());
    for (int i=0;i<(int)localSlots.size();i++) {
      assert(localSlots[i].dataRank >= 0 ||
             i == 0 && localSlots[i].dataRank == -1);
      assert(!localSlots[i].gpuIDs.empty());
    }
    
    if (!isActiveWorker)  {
      // not an active worker: no device groups etc, just create a
      // single default device
      throw std::runtime_error
        ("inactive workers not implemented right now");
      return;
    }


    //   throw std::runtime_error("not enough GPUs ("
    //                            +std::to_string(gpuIDs.size())
    //                            +") for requested num data groups ("
    //                            +std::to_string(dataGroupIDs.size())
    //                            +")");
    // if (gpuIDs.size() % dataGroupIDs.size())
    //   throw std::runtime_error("requested num GPUs is not a multiple of "
    //                            "requested num data groups");
    // int numSlots = (int)dataGroupIDs.size();
    // int gpusPerSlot = (int)gpuIDs.size() / numSlots;
    // std::vector<std::vector<int>> gpuInSlot(numSlots);
    // perSlot.resize(numSlots);
    
    std::vector<Device *> allLocalDevices;
    // std::vector<WorkerTopo::Device> allDevices;

    int numSlots = (int)localSlots.size();
    perSlot.resize(numSlots);

    havePeerAccess = true;
    for (int lmsIdx=0;lmsIdx<numSlots;lmsIdx++) {
      auto &ls = localSlots[lmsIdx];
      auto &dg = perSlot[lmsIdx];
      dg.context = this;
      dg.modelRankInThisSlot = ls.dataRank;
      havePeerAccess
        = havePeerAccess & rtc::enablePeerAccess(ls.gpuIDs);

      std::vector<Device *> slotDevices;
      for (auto gpuID : ls.gpuIDs) {
        rtc::Device *rtc = new rtc::Device(gpuID);
        int nextLocal = allLocalDevices.size();
        Device *device 
          = new Device(rtc,topo.get(),nextLocal);
          
        slotDevices.push_back(device);
        allLocalDevices.push_back(device);
        dg.gpuIDs.push_back(gpuID);
      }
      dg.devices
        = std::make_shared<DevGroup>(slotDevices,(int)allLocalDevices.size());
    }

    devices = std::make_shared<DevGroup>
      (allLocalDevices,(int)allLocalDevices.size());
    if (!havePeerAccess) {
      std::cout << "don't have peer access between GPUs ... this is going to get interesting" << std::endl;
      deviceWeNeedToCopyToForFBMap = allLocalDevices[0];
    }
    
    for (auto &dg : perSlot)
      dg.materialRegistry
        = std::make_shared<render::MaterialRegistry>(dg.devices);
    for (auto &dg : perSlot)
      dg.samplerRegistry
        = std::make_shared<render::SamplerRegistry>(dg.devices);
  }
  
  Context::~Context()
  {
    hostOwnedHandles.clear();

    perSlot.clear();
    for (auto &device : *devices) {
      delete device;
      device = 0;
    }
  }

  /*! returns how many rays are active in all ray queues, across all
    devices and, where applicable, across all ranks. We use this to
    decide whether we can terminate a frame, or need more bounces - we
    may actually need to enter another bounce even if *we* do not have
    any rays */
  int Context::numRaysActiveLocally()
  {
    int numActive = 0;
    for (auto device : *devices)
      numActive += device->rayQueue->numActiveRays();
    return numActive;
  }
  
  
  void Context::finalizeTiles(FrameBuffer *fb)
  {
    fb->finalizeTiles();
  }

  void Context::renderTiles(Renderer    *renderer,
                            GlobalModel *model,
                            Camera      *camera,
                            FrameBuffer *fb)
  {
    auto _context = this;
    if (!isActiveWorker)
      return;

    for (auto device : *devices)
      device->syncPipelineAndSBT();

    // iw - todo: add wave-front-merging here.
    for (int p=0;p<renderer->pathsPerPixel;p++) {

      if (FromEnv::get()->logQueues) 
        std::cout << "#################### RENDER ######################" << std::endl;
      // double _t0 = getCurrentTime();
      if (FromEnv::get()->logQueues) 
        std::cout << "==================== new pixel wave ======================" << std::endl;
      generateRays(camera,renderer,fb);
      for (int generation=0;true;generation++) {
        if (FromEnv::get()->logQueues) 
          std::cout << "-------------------- new generation " << generation << " ----------------------" << std::endl;

        bool needHitIDs = fb->needHitIDs() && (generation==0);
        uint32_t rngSeed = fb->accumID*16+generation;
        if (FromEnv::get()->logQueues) 
          std::cout << "----- trace (glob) " << generation
                    << " -----------" << std::endl;
        traceRaysGlobally(model,rngSeed,needHitIDs);

        if (FromEnv::get()->logQueues) 
          std::cout << "----- shade " << generation
                    << " -----------" << std::endl;
        shadeRaysLocally(renderer, model, fb, generation, rngSeed);
        // no sync required here, shadeRays syncs itself.

        if (FromEnv::get()->logQueues) 
          std::cout << "----- shade " << generation
                    << " -----------" << std::endl;
        const int numActiveGlobally = numRaysActiveGlobally();
        if (FromEnv::get()->logQueues)
          printf("#generation %i num active %s after bounce\n",
                 generation,prettyNumber(numActiveGlobally).c_str());
        if (numActiveGlobally > 0)
          continue;
    
        break;
      }
      ++ fb->accumID;
    }
  }

    
  /*! trace all rays currently in a ray queue, including forwarding
    if and where applicable, untile every ray in the ray queue as
    found its intersection */
  void Context::traceRaysGlobally(GlobalModel *model, uint32_t rngSeed, bool needHitIDs)
  {
    // if (myRank() == 0) printf("globaltrace....\n");
    
    if (FromEnv::get()->logQueues) 
      printf("(mr%i) traceRaysGlobally\n",myRank());
    globalTraceImpl->traceRays(model,rngSeed,needHitIDs);
  }

  std::shared_ptr<barney_api::Model> Context::createModel()
  {
    return GlobalModel::create(this);
  }
  
  std::shared_ptr<barney_api::Renderer> Context::createRenderer()
  {
    return Renderer::create(this);
  }

  void Context::ensureRayQueuesLargeEnoughFor(FrameBuffer *fb)
  {
    if (!isActiveWorker)
      return;

    auto dev0 = (*devices)[0];
    auto devFB = fb->getFor(dev0);
    int numTilesInFrame        = devFB->numTiles.x*devFB->numTiles.y;
    int numGPUsThatRenderTiles = topo->numWorkerDevices;
    int maxTilesOnAnyGPU       = divRoundUp(numTilesInFrame,
                                            numGPUsThatRenderTiles);
    int upperBoundOnNumRays
      = maxTilesOnAnyGPU * /* max two rays per pixel*/2 * BARNEY_NS::pixelsPerTile;
    for (auto device : *devices) {
      assert(device->rayQueue);
      device->rayQueue->resize(upperBoundOnNumRays);
    }
    
  }

  int Context::contextSize() const
  {
    return (int)devices->size();
  }
  
  SlotContext *Context::getSlot(int slot)
  {
    assert(slot >= 0);
    assert(slot < perSlot.size());
    return &perSlot[slot];
  }

  bool Context::logging() 
  {
#ifdef NDEBUG
    return false;
#else
    return true;
#endif
  }
  
  /*! helper function to print a warning when app tries to create anari
    object of certain kind and type that barney does not support */
  void Context::warn_unsupported_object(const std::string &kind,
                                        const std::string &type)
  {
    static std::set<std::string> alreadyWarned;
    if (alreadyWarned.find(kind+"::"+type) != alreadyWarned.end())
      return;
    std::cout << OWL_TERMINAL_RED
              << "#bn: asked to create object of unknown/unsupported "
              <<  kind << " of type '" << type << "'"
              << " that I know nothing about"
              << OWL_TERMINAL_DEFAULT << std::endl;
    alreadyWarned.insert(kind+"::"+type);
  }

  std::shared_ptr<barney_api::Camera>
  Context::createCamera(const std::string &type)
  {
    return Camera::create(this,type);
  }
  
  std::shared_ptr<barney_api::Volume>
  Context::createVolume(const std::shared_ptr<barney_api::ScalarField> &sf)
  {
    return Volume::create(sf->as<ScalarField>());
  }

  std::shared_ptr<barney_api::TextureData> 
  Context::createTextureData(int slot,
                             BNDataType texelFormat,
                             vec3i dims,
                             const void *texels)
  {
    return std::make_shared<TextureData>(this,
                                         getDevices(slot),
                                         texelFormat,
                                         dims,texels);
  }

  std::shared_ptr<barney_api::Texture>
  Context::createTexture(const std::shared_ptr<barney_api::TextureData> &td,
                         BNTextureFilterMode  filterMode,
                         BNTextureAddressMode addressModes[],
                         BNTextureColorSpace  colorSpace)
  {
    return std::make_shared<Texture>(this,
                                     td->as<TextureData>(),
                                     filterMode,addressModes,colorSpace);
  }
  
    
  std::shared_ptr<barney_api::ScalarField>
  Context::createScalarField(int slot, const std::string &type)
  {
    return ScalarField::create(this,getDevices(slot),type);
  }
    
  std::shared_ptr<barney_api::Geometry>
  Context::createGeometry(int slot, const std::string &type) 
  {
    return Geometry::create(this,getDevices(slot),type);
  }
    
  std::shared_ptr<barney_api::Material>
  Context::createMaterial(int slot, const std::string &type) 
  {
    return render::HostMaterial::create(getSlot(slot),type);
  }

  std::shared_ptr<barney_api::Sampler>
  Context::createSampler(int slot, const std::string &type) 
  {
    return render::Sampler::create(getSlot(slot),type);
  }

  std::shared_ptr<barney_api::Light>
  Context::createLight(int slot, const std::string &type) 
  {
    return Light::create(this,getDevices(slot),type);
  }

  std::shared_ptr<barney_api::Group>
  Context::createGroup(int slot,
                       barney_api::Geometry **_geoms, int numGeoms,
                       barney_api::Volume **_volumes, int numVolumes) 
  {
    std::vector<Geometry::SP> geoms;
    std::vector<Volume::SP> volumes;
    for (int i=0;i<numGeoms;i++) {
      auto g = _geoms[i];
      if (g) geoms.push_back(g->as<Geometry>());
    }
    for (int i=0;i<numVolumes;i++) {
      auto g = _volumes[i];
      if (g) volumes.push_back(g->as<Volume>());
    }
    return std::make_shared<Group>(this,
                                   getDevices(slot),
                                   geoms,volumes);
  }

  std::shared_ptr<barney_api::Data>
  Context::createData(int slot,
                      BNDataType dataType)
  {
    return BaseData::create(this,getDevices(slot),dataType);
  }
  
} // ::BARNEY_NS

