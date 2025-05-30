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

#pragma once

#include "barney/api/Context.h"
#include "barney/Object.h"
#include <set>

namespace BARNEY_NS {
  using namespace owl::common;
  using barney_api::FromEnv;
  
  struct FrameBuffer;
  struct GlobalModel;
  struct Camera;
  struct Renderer;
  struct Geometry;
  
  namespace render {
    struct HostMaterial;
    struct SamplerRegistry;
    struct MaterialRegistry;
    struct DeviceMaterial;    
  };

  struct SlotContext {
    Context *context;
    int modelRankInThisSlot;//dataGroupID;
    /*! device(s) inside this data group; will be a subset of
      Context::devices */
    std::vector<int>     gpuIDs;
    DevGroup::SP         devices;
    std::shared_ptr<render::HostMaterial>     defaultMaterial = 0;
    std::shared_ptr<render::SamplerRegistry>  samplerRegistry = 0;
    std::shared_ptr<render::MaterialRegistry> materialRegistry = 0;
  };
  
  struct Context : public barney_api::Context
  {
    Context(const std::vector<int> &dataGroupIDs,
            const std::vector<int> &gpuIDs,
            int globalIndex,
            int globalIndexStep);
    virtual ~Context();
    
    /*! create a frame buffer object suitable to this context */
    // virtual FrameBuffer *createFB(int owningRank) = 0;
    std::shared_ptr<barney_api::Model>
    createModel() override;
    
    std::shared_ptr<barney_api::Renderer>
    createRenderer() override;

    std::shared_ptr<barney_api::Volume>
    createVolume(const std::shared_ptr<barney_api::ScalarField> &sf) override;

    std::shared_ptr<barney_api::TextureData> 
    createTextureData(int slot,
                      BNDataType texelFormat,
                      vec3i dims,
                      const void *texels) override;
    
    std::shared_ptr<barney_api::ScalarField>
    createScalarField(int slot, const std::string &type) override;
    
    std::shared_ptr<barney_api::Geometry>
    createGeometry(int slot, const std::string &type) override;
    
    std::shared_ptr<barney_api::Material>
    createMaterial(int slot, const std::string &type) override;

    std::shared_ptr<barney_api::Sampler>
    createSampler(int slot, const std::string &type) override;

    std::shared_ptr<barney_api::Light>
    createLight(int slot, const std::string &type) override;

    std::shared_ptr<barney_api::Group>
    createGroup(int slot,
                barney_api::Geometry **geoms, int numGeoms,
                barney_api::Volume **volumes, int numVolumes) override;

    std::shared_ptr<barney_api::Data>
    createData(int slot,
               BNDataType dataType,
               size_t numItems,
               const void *items) override;
    
    std::shared_ptr<barney_api::Camera>
    createCamera(const std::string &type) override;

    std::shared_ptr<barney_api::Texture>
    createTexture(const std::shared_ptr<barney_api::TextureData> &td,
                  BNTextureFilterMode  filterMode,
                  BNTextureAddressMode addressModes[],
                  BNTextureColorSpace  colorSpace) override;
 
    
    // std::shared_ptr<render::HostMaterial> getDefaultMaterial(int slot);

    static bool logging();
    
    /*! pretty-printer for printf-debugging */
    virtual std::string toString() const 
    { return "<Context(abstract)>"; }

    render::World *getWorld(int slot);
    
    /* goes across all devices, syncs that device, and checks for
       errors - careful, this will be very slow, shoudl only be used
       for debugging multi-gpu race conditions and such */
    void syncCheckAll(const char *where="");
    
    // for debugging ...
    virtual void barrier(bool warn=true) {}
    
    /*! generate a new wave-front of rays */
    void generateRays(Camera *camera,
                      Renderer *renderer,
                      FrameBuffer *fb);
    
    /*! have each *local* GPU trace its current wave-front of rays */
    void traceRaysLocally(GlobalModel *model, uint32_t rngSeed, bool needHitIDs);
    
    /*! trace all rays currently in a ray queue, including forwarding
      if and where applicable, untile every ray in the ray queue as
      found its intersection */
    void traceRaysGlobally(GlobalModel *model, uint32_t rngSeed, bool needHitIDs);

    /*! forward rays (during global trace); returns if _after_ that
      forward the rays need more tracing (true) or whether they're
      done (false) */
    virtual bool forwardRays(bool needHitIDs) = 0;

    /*! returns how many rays are active in all ray queues, across all
      devices and, where applicable, across all ranks */
    int numRaysActiveLocally();

    /*! returns how many rays are active in all ray queues, across all
      devices and, where applicable, across all ranks */
    virtual int numRaysActiveGlobally() = 0;

    void shadeRaysLocally(Renderer *renderer,
                          GlobalModel *model,
                          FrameBuffer *fb,
                          int generation,
                          uint32_t rngSeed);
    void finalizeTiles(FrameBuffer *fb);
    
    void renderTiles(Renderer *renderer,
                     GlobalModel *model,
                     Camera      *camera,
                     FrameBuffer *fb);
    
    virtual void render(Renderer    *renderer,
                        GlobalModel *model,
                        Camera      *camera,
                        FrameBuffer *fb) = 0;

    void ensureRayQueuesLargeEnoughFor(FrameBuffer *fb);

    /*! helper function to print a warning when app tries to create an
        object of certain kind and type that barney does not
        support */
    void warn_unsupported_object(const std::string &kind,
                                 const std::string &type);
    DevGroup::SP getDevices(int slot)
    {
      if (slot < 0)
        return this->devices;
      else 
        return getSlot(slot)->devices;
    }
    
    int contextSize() const;

    const bool isActiveWorker;

    /*! whether we have successfully enabled peer access across all
        GPUs (eg, to allow tiledFB to write to gpu 0 linear fb */
    bool havePeerAccess = false;
    
    SlotContext *getSlot(int slot);
    std::vector<SlotContext> perSlot;
    DevGroup::SP devices;
    int const globalIndex;
  };
  

  // ==================================================================
  // INLINE IMPLEMENTATION SECTION
  // ==================================================================


  /* goes across all devices, syncs that device, and checks for
     errors - careful, this will be very slow, shoudl only be used
     for debugging multi-gpu race conditions and such */
  inline void Context::syncCheckAll(const char *where)
  {
    for (auto device : *devices)
      device->sync();
  }
    
}

