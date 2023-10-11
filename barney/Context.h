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

#include "barney.h"
#include "mori/Ray.h"
#include "mori/Camera.h"
#include "mori/MoriContext.h"
#include "mori/cuda-helper.h"
#include "mori/TiledFB.h"
#include <string.h>
#include <cuda_runtime.h>
#include <mutex>
#include <map>

namespace barney {
  using namespace owl::common;
  using mori::SetActiveGPU;
  
  struct Object {
    typedef std::shared_ptr<Object> SP;

    /*! pretty-printer for printf-debugging */
    virtual std::string toString() const
    { return "<Object>"; }
  };

  struct FrameBuffer;
  struct Model;

  struct Context;
  
  struct DeviceContext : public mori::MoriContext {
    Context *barney = 0;
    
    struct { int rank = -1; int gpu = -1; } next, prev;
  };

  struct Context : public Object {
    
    Context(const std::vector<int> &dataGroupIDs,
            const std::vector<int> &gpuIDs);
    ~Context()
    { for (auto devCon : perGPU) delete devCon; }
    
    /*! create a frame buffer object suitable to this context */
    virtual FrameBuffer *createFB(int owningRank) = 0;
    Model *createModel();
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "<Context(abstract)>"; }

    template<typename T>
    T *initReference(std::shared_ptr<T> sp)
    {
      std::lock_guard<std::mutex> lock(mutex);
      hostOwnedHandles[sp]++;
      return sp.get();
    }

    /*! generate a new wave-front of rays */
    void generateRays(const mori::Camera &camera,
                      FrameBuffer *fb);
    
    /*! have each *local* GPU trace its current wave-front of rays */
    void traceRaysLocally();
    
    /*! trace all rays currently in a ray queue, including forwarding
        if and where applicable, untile every ray in the ray queue as
        found its intersection */
    void traceRaysGlobally();

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

    void shadeRaysLocally(FrameBuffer *fb);
    void finalizeTiles(FrameBuffer *fb);
    
    void renderTiles(Model *model,
                     const mori::Camera &camera,
                     FrameBuffer *fb);
    
    virtual void render(Model *model,
                        const mori::Camera *camera,
                        FrameBuffer *fb) = 0;
    
    const std::vector<int> dataGroupIDs;
    const std::vector<int> gpuIDs;

    std::mutex mutex;
    std::map<Object::SP,int> hostOwnedHandles;
    std::vector<mori::DeviceGroup::SP> moris;

    void ensureRayQueuesLargeEnoughFor(FrameBuffer *fb);
    
    std::vector<DeviceContext *> perGPU;
    const bool isActiveWorker;
  };
  
  // /*! TEMP function - will die pretty soon */
  // void renderTiles_testFrame(Context *context,
  //                            int localID,
  //                            Model *model,
  //                            FrameBuffer *fb,
  //                            const BNCamera *camera);
  // /*! TEMP function - will die pretty soon */
  // void renderTiles_rayDir(Context *context,
  //                         int localID,
  //                         Model *model,
  //                         FrameBuffer *fb,
  //                         const BNCamera *camera);
  // /*! TEMP function - will die pretty soon */
  // void renderTiles(Context *context,
  //                  int localID,
  //                  Model *model,
  //                  FrameBuffer *fb,
  //                  const BNCamera *camera);
  
}

