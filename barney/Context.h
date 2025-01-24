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

#include "barney/render/Ray.h"
#include "barney/geometry/Geometry.h"
#include "barney/Camera.h"
// #include "barney/DeviceContext.h"
#include "barney/common/cuda-helper.h"
#include "barney/fb/TiledFB.h"
#include "barney/Object.h"
#include "barney/render/Renderer.h"
#include <set>
#include "barney/render/RayQueue.h"

namespace barney {
  using namespace owl::common;
  using render::Ray;
  
  struct FrameBuffer;
  struct GlobalModel;

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
  
  struct Context// : public Object
  {
    Context(const std::vector<int> &dataGroupIDs,
            const std::vector<int> &gpuIDs,
            int globalIndex,
            int globalIndexStep);
    virtual ~Context();
    
    /*! create a frame buffer object suitable to this context */
    virtual FrameBuffer *createFB(int owningRank) = 0;
    GlobalModel *createModel();
    Renderer *createRenderer();

    // std::shared_ptr<render::HostMaterial> getDefaultMaterial(int slot);

    static bool logging();
    
    /*! pretty-printer for printf-debugging */
    virtual std::string toString() const 
    { return "<Context(abstract)>"; }

    template<typename T>
    T *initReference(std::shared_ptr<T> sp)
    {
      if (!sp) return 0;
      std::lock_guard<std::mutex> lock(mutex);
      hostOwnedHandles[sp]++;
      return sp.get();
    }
    
    /*! decreases (the app's) reference count of said object by
      one. if said refernce count falls to 0 the object handle gets
      destroyed and may no longer be used by the app, and the object
      referenced to by this handle may be removed (from the app's
      point of view). Note the object referenced by this handle may
      not get destroyed immediagtely if it had other indirect
      references, such as, for example, a group still holding a
      refernce to a geometry */
    void releaseHostReference(Object::SP object);
    
    /*! increases (the app's) reference count of said object byb
        one */
    void addHostReference(Object::SP object);

    render::World *getWorld(int slot);
    
    /* goes across all devices, syncs that device, and checks for
       errors - careful, this will be very slow, shoudl only be used
       for debugging multi-gpu race conditions and such */
    void syncCheckAll(const char *where="");
    
    // for debugging ...
    virtual void barrier(bool warn=true) {}
    
    /*! generate a new wave-front of rays */
    void generateRays(const barney::Camera::DD &camera,
                      Renderer *renderer,
                      FrameBuffer *fb);
    
    /*! have each *local* GPU trace its current wave-front of rays */
    void traceRaysLocally(GlobalModel *model);
    
    /*! trace all rays currently in a ray queue, including forwarding
      if and where applicable, untile every ray in the ray queue as
      found its intersection */
    void traceRaysGlobally(GlobalModel *model);

    /*! forward rays (during global trace); returns if _after_ that
      forward the rays need more tracing (true) or whether they're
      done (false) */
    virtual bool forwardRays() = 0;

    /*! returns how many rays are active in all ray queues, across all
      devices and, where applicable, across all ranks */
    int numRaysActiveLocally();

    /*! returns how many rays are active in all ray queues, across all
      devices and, where applicable, across all ranks */
    virtual int numRaysActiveGlobally() = 0;

    void shadeRaysLocally(Renderer *renderer,
                          GlobalModel *model,
                          FrameBuffer *fb,
                          int generation);
    void finalizeTiles(FrameBuffer *fb);
    
    void renderTiles(Renderer *renderer,
                     GlobalModel *model,
                     const barney::Camera::DD &camera,
                     FrameBuffer *fb);
    
    virtual void render(Renderer    *renderer,
                        GlobalModel *model,
                        const barney::Camera::DD &camera, 
                        FrameBuffer *fb) = 0;

    std::mutex mutex;
    std::map<Object::SP,int> hostOwnedHandles;
    // std::vector<barney::DeviceGroup::SP> barneys;

    void ensureRayQueuesLargeEnoughFor(FrameBuffer *fb);


    
    /*! helper function to print a warning when app tries to create an
        object of certain kind and type that barney does not
        support */
    void warn_unsupported_object(const std::string &kind,
                                 const std::string &type);
    std::set<std::string> alreadyWarned;

    SlotContext *getSlot(int slot);
    std::vector<SlotContext> perSlot;
    
    const bool isActiveWorker;

    /*! return the single 'global' own context that spans all gpus, no
        matter how many model slots those are grouped in; in theory
        this should be used for very, very little - almost all data
        should live in a model slots (which has its own context), only
        truly global data (such as renderer background image) should
        ever be global */
    // OWLContext getOWL(int slot);// { return globalContextAcrossAllGPUs; }
    // OWLContext getGlobalOWL() const;
    // rtc::DevGroup *getRTC(int slot) const;
    
    
    DevGroup::SP getDevices(int slot) {
      if (slot < 0)
        return this->devices;
      else 
        return getSlot(slot)->devices;
    }
    
    int contextSize() const;

    DevGroup::SP devices;
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
      device->rtc->sync();
      // {
      // SetActiveGPU forDuration(dev->device);

      // cudaDeviceSynchronize();                                   
      // cudaError_t rc = cudaGetLastError();                        
      // if (rc != cudaSuccess) {                                    
      //   printf("******************************************************************\nCUDA fatal error %s (%s)\n",
      //          cudaGetErrorString(rc),where);
      //   fflush(0);
      //   throw std::runtime_error("unrecoverable cuda error");
      // }                                                              
  // }
  }
    
}

