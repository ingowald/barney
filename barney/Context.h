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
  
  struct DeviceContext : public mori::DeviceContext {
    Context *barney = 0;
    mori::RayQueue rays;
    struct { int rank = -1; int gpu = -1; } next, prev;
  };

  struct Context : public Object {
    
    Context(const std::vector<int> &dataGroupIDs,
            const std::vector<int> &gpuIDs);
    ~Context()
    { for (auto devCon : deviceContexts) delete devCon; }
    
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

    virtual void render(Model *model,
                        const BNCamera *camera,
                        FrameBuffer *fb) = 0;
    
    const std::vector<int> dataGroupIDs;
    const std::vector<int> gpuIDs;

    std::mutex mutex;
    std::map<Object::SP,int> hostOwnedHandles;
    std::vector<mori::DeviceGroup::SP> moris;

    void ensureRayQueuesLargeEnoughFor(vec2i fbSize);
    size_t currentRayQueueSize = 0;
    
    std::vector<DeviceContext *> deviceContexts;
    const bool isActiveWorker;
  };
  
  /*! TEMP function - will die pretty soon */
  void renderTiles_testFrame(Context *context,
                             int localID,
                             Model *model,
                             FrameBuffer *fb,
                             const BNCamera *camera);
  /*! TEMP function - will die pretty soon */
  void renderTiles_rayDir(Context *context,
                          int localID,
                          Model *model,
                          FrameBuffer *fb,
                          const BNCamera *camera);
  /*! TEMP function - will die pretty soon */
  void renderTiles(Context *context,
                   int localID,
                   Model *model,
                   FrameBuffer *fb,
                   const BNCamera *camera);
  
}

