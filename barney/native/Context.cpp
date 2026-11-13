// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/Context.h"
#include "native/DeviceGroup.h"
#include "native/fb/FrameBuffer.h"
// #include "native/GlobalModel.h"
#include "native/render/RayQueue.h"
#include "native/render/Sampler.h"
#include "native/render/SamplerRegistry.h"
#include "native/render/MaterialRegistry.h"
#include "native/Camera.h"
#include "native/render/Renderer.h"
#include "native/FromEnv.h"
#include "native/volume/Volume.h"
#include "native/geometry/Geometry.h"
#include "native/Group.h"

namespace BARNEY_NS {
  namespace native {
    
    Context::Context(const std::vector<int> &gpuIDs,
                     const std::vector<int> &dataGroupIDs,
                     WorkerTopo::SP topo)
      : topo(topo)
    {
      // assert(!localDataGroups.empty());
      assert(gpuIDs.size() == dataGroupIDs.size());
      assert(!gpuIDs.empty());
      // for (auto &ldg : localDataGroups) {
      //   assert(!ldg.gpuIDs.empty());
      //   for (auto gpu : ldg.gpuIDs)
      //     assert(gpu >= 0);
      // }
      
      // assert(!localDataGroups.empty());
      // for (int i=0;i<(int)localDataGroups.size();i++) {
      //   assert(localDataGroups[i].dataRank >= 0 ||
      //          i == 0 && localDataGroups[i].dataRank == -1);
      //   assert(!localDataGroups[i].gpuIDs.empty());
      // }
    
      // if (!isActiveWorker)  {
      //   // not an active worker: no device groups etc, just create a
      //   // single default device
      //   throw std::runtime_error
      //     ("inactive workers not implemented right now");
      //   return;
      // }

      // { // try to enable peer access across all GPUs across all of
      //   // this context's data groups
      //   havePeerAccess = true;
      //   // std::vector<int> allGPUs;
      //   // for (auto &dg : localDataGroups)
      //   //   for (auto gpu : dg.gpuIDs)
      //   //     allGPUs.push_back(gpu);
      //   havePeerAccess
      //     = havePeerAccess && rtc::enablePeerAccess(gpuIDs);//allGPUs);
      // }
      havePeerAccess = rtc::enablePeerAccess(gpuIDs);//allGPUs);
      
      std::vector<Device *> allLocalDevices;
      int numDevices = gpuIDs.size();//numLDGs = (int)localDataGroups.size();
      // for (int i=0;i<numLDGs;i++)
      //   perLDG.push_back(new LDGContext);

      for (int localDevIdx=0;localDevIdx<numDevices;localDevIdx++) {
      // for (int ldgIdx=0;ldgIdx<numLDGs;ldgIdx++) {
      //   auto &ldgSpec = localDataGroups[ldgIdx];
      //   auto &dg = *perLDG[ldgIdx];
        // dg.context = this;
        // dg.deviceIdx = ldgIdx;
        // dg.modelRankInThisSlot = ldgSpec.dataRank;

        // std::vector<Device *> devicesForThisLDG;
        // for (auto gpuID : ldgSpec.gpuIDs) {
        int gpuID = gpuIDs[localDevIdx];
        int dgID = dataGroupIDs[localDevIdx];
        rtc::Device *rtc = new rtc::Device(gpuID);
        // int localDeviceRank = (int)allLocalDevices.size();
        Device *device 
          = new Device(rtc,topo.get(),this,localDevIdx);//localDeviceRank);
          
        // devicesForThisLDG.push_back(device);
        allLocalDevices.push_back(device);
        // }
        // dg.devices
        //   = std::make_shared<DevGroup>(devicesForThisLDG,
        //                                (int)allLocalDevices.size());
      }
      devices = allLocalDevices;
      // devices = std::make_shared<DevGroup>
      //   (allLocalDevices,(int)allLocalDevices.size());
      if (!havePeerAccess) {
        std::cout << "don't have peer access between GPUs ... this is going to get interesting" << std::endl;
        deviceWeNeedToCopyToForFBMap = allLocalDevices[0];
      }
    
      // for (auto &dg : perLDG) {
      //   dg->materialRegistry
      //     = std::make_shared<MaterialRegistry>(dg->devices);
      //   dg->samplerRegistry
      //     = std::make_shared<SamplerRegistry>(dg->devices);
      // }
    }
  
    Context::~Context()
    {
      hostOwnedHandles.clear();

      delete globalTraceImpl;
      globalTraceImpl = 0;

      // for (auto dg : perLDG) delete dg;
      // perLDG.clear();
      
      for (auto &device : devices) {
        delete device;
        device = 0;
      }
    }

     int Context::myRank()
     { return 0; }
    
     int Context::mySize()
     { return 1; }

    /*! returns how many rays are active in all ray queues, across all
      devices and, where applicable, across all ranks. We use this to
      decide whether we can terminate a frame, or need more bounces - we
      may actually need to enter another bounce even if *we* do not have
      any rays */
    int Context::numRaysActiveLocally()
    {
      int numActive = 0;
      for (auto device : devices)
        numActive += device->rayQueue->numActiveRays();
      return numActive;
    }
  
  
    void Context::finalizeTiles(FrameBuffer *fb)
    {
      fb->finalizeTiles();
    }

    void Context::renderTiles(const std::vector<Renderer*> &renderers,
                              const std::vector<World *> &worlds,
                              Camera      *camera,
                              FrameBuffer *fb)
    {
      auto _context = this;
      // if (!isActiveWorker)
      //   return;

      for (auto device : devices)
        device->syncPipelineAndSBT();

      activeCutPlane = renderers[0]->cutPlane;

      // iw - todo: add wave-front-merging here.
      int ppp = renderers[0]->pathsPerPixel;
      for (int p=0;p<ppp;p++) {

        if (FromEnv::logQueues) 
          std::cout << "#################### RENDER ######################" << std::endl;
        if (FromEnv::logQueues) 
          std::cout << "==================== new pixel wave ======================" << std::endl;
        generateRays(camera,renderers,fb);
        for (int generation=0;true;generation++) {
          if (FromEnv::logQueues) 
            std::cout << "-------------------- new generation " << generation << " ----------------------" << std::endl;

          bool needHitIDs = fb->needHitIDs() && (generation==0);
          uint32_t rngSeed = fb->accumID*16+generation;
          traceRaysGlobally(worlds,rngSeed,needHitIDs);

          shadeRaysLocally(renderers, worlds, camera, fb, generation, rngSeed);

          const int numActiveGlobally = numRaysActiveGlobally();
          if (FromEnv::logQueues)
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
    void Context::traceRaysGlobally(const std::vector<World *> &worlds,
                                    uint32_t rngSeed,
                                    bool needHitIDs)
    {
      // if (myRank() == 0) printf("globaltrace....\n");
      if (FromEnv::logQueues) 
        printf("(mr%i) traceRaysGlobally\n",myRank());
      globalTraceImpl->traceRays(worlds,rngSeed,needHitIDs);
    }

    // std::shared_ptr<GlobalModel> Context::createModel()
    // {
    //   return GlobalModel::create(this);
    // }
  
    std::shared_ptr<Renderer> Context::createRenderer(int slot)
    {
      return std::make_shared<Renderer>(devices[slot]);
    }

    void Context::ensureRayQueuesLargeEnoughFor(FrameBuffer *fb)
    {
      // if (!isActiveWorker)
      //   return;

      auto dev0 = devices[0];
      auto devFB = fb->getFor(dev0);
      int numTilesInFrame        = devFB->numTiles.x*devFB->numTiles.y;
      int numGPUsThatRenderTiles = topo->numWorkerDevices;
      int maxTilesOnAnyGPU       = divRoundUp(numTilesInFrame,
                                              numGPUsThatRenderTiles);
      int upperBoundOnNumRays
        = maxTilesOnAnyGPU
        * /* max two rays per pixel*/2
        * BARNEY_NS::native::pixelsPerTile;
      for (auto device : devices) {
        assert(device->rayQueue);
        device->rayQueue->resize(upperBoundOnNumRays);
      }
    
    }

    // int Context::contextSize() const
    // {
    //   return (int)devices->size();
    // }
  
    // LDGContext *Context::getLDG(int localDataGroupIndex)
    // {
    //   assert(localDataGroupIndex >= 0);
    //   assert(localDataGroupIndex < perLDG.size());
    //   return perLDG[localDataGroupIndex];
    // }

    bool Context::logging() 
    {
#ifdef NDEBUG
      return false;
#else
      return true;
#endif
    }


    void Context::releaseHostReference(Object *object)
    {
      assert(object);
      std::lock_guard<std::mutex> lock(mutex);
      auto it = hostOwnedHandles.find(object);
      if (it == hostOwnedHandles.end())
        throw std::runtime_error
          ("trying to bnRelease() a handle that either does not "
           "exist, or that the app (no lnoger) has any valid references on");
      
      hostOwnedHandles.erase(it);
    }
  
    // /*! helper function to print a warning when app tries to create anari
    //   object of certain kind and type that barney does not support */
    // void Context::warn_unsupported_object(const std::string &kind,
    //                                       const std::string &type)
    // {
    //   static std::set<std::string> alreadyWarned;
    //   if (alreadyWarned.find(kind+"::"+type) != alreadyWarned.end())
    //     return;
    //   std::cout << OWL_TERMINAL_RED
    //             << "#bn: asked to create object of unknown/unsupported "
    //             <<  kind << " of type '" << type << "'"
    //             << " that I know nothing about"
    //             << OWL_TERMINAL_DEFAULT << std::endl;
    //   alreadyWarned.insert(kind+"::"+type);
    // }

    std::shared_ptr<Camera>
    Context::createCamera(const std::string &type)
    {
      return Camera::create(devices[0],type);
    }
  
    std::shared_ptr<Volume>
    Context::createVolume(const std::shared_ptr<SpatialField> &sf)
    {
      // sf->as<SpatialField>()
      if (!sf) return {};
      return Volume::create(sf);
    }

    std::shared_ptr<TextureData> 
    Context::createTextureData(int slot,
                               BNDataType texelFormat,
                               vec3i dims,
                               const void *texels)
    {
      return std::make_shared<TextureData>(devices[slot],
                                           // this,
                                           // getDevices(slot),
                                           texelFormat,
                                           dims,texels);
    }

    std::shared_ptr<Texture>
    Context::createTexture(const std::shared_ptr<TextureData> &td,
                           BNTextureFilterMode  filterMode,
                           BNTextureAddressMode addressModes[],
                           BNTextureColorSpace  colorSpace)
    {
      return std::make_shared<Texture>(td->as<TextureData>(),
                                       filterMode,addressModes,colorSpace);
    }
  
    
    std::shared_ptr<SpatialField>
    Context::createSpatialField(int slot, const std::string &type)
    {
      return SpatialField::create(devices[slot],type);
    }
    
    std::shared_ptr<Geometry>
    Context::createGeometry(int slot, const std::string &type) 
    {
      return Geometry::create(devices[slot],type);
    }
    
    std::shared_ptr<HostMaterial>
    Context::createMaterial(int slot, const std::string &type) 
    {
      return HostMaterial::create(devices[slot],
                                  // getLDG(slot),
                                  type);
    }

    std::shared_ptr<Sampler>
    Context::createSampler(int slot, const std::string &type) 
    {
      return Sampler::create(devices[slot],//getLDG(slot),
                             type);
    }

    std::shared_ptr<Light>
    Context::createLight(int slot, const std::string &type) 
    {
      return Light::create(devices[slot],type);
    }

    std::shared_ptr<Group>
    Context::createGroup(int slot,
                         Geometry **_geoms, int numGeoms,
                         Volume **_volumes, int numVolumes) 
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
      return std::make_shared<Group>(devices[slot],
                                     // this,
                                     // getDevices(slot),
                                     geoms,volumes);
    }

    std::shared_ptr<Data>
    Context::createData(int slot,
                        BNDataType dataType)
    {
      return BaseData::create(devices[slot],dataType);
    }
    
    // DevGroup::SP Context::getDevices(int slot)
    // {
    //   if (slot < 0)
    //     return this->devices;
    //   else 
    //     return getLDG(slot)->devices;
    // }

    void *Context::initReference(Object *object)
    {
      if (!object) return 0;
      std::lock_guard<std::mutex> lock(mutex);
      hostOwnedHandles[object] = object->shared_from_this();
      return object;
    }
    
  }
} // ::BARNEY_NS

