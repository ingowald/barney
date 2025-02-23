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

#include "barney/render/OptixGlobals.h"
#include "barney/Context.h"
#include "barney/GlobalModel.h"
#include "barney/ModelSlot.h"
#include "barney/render/SamplerRegistry.h"
#include "barney/render/MaterialRegistry.h"
#include "rtcore/RTCore.h"


// __constant__ barney::render::OptixGlobals optixLaunchParams;
// // DECLARE_OPTIX_LAUNCH_PARAMS(barney::render::OptixGlobals);

namespace BARNEY_NS {
  RTC_DECLARE_GLOBALS(render::OptixGlobals);
  
  namespace render {

    struct TraceRays {
      inline __device__ static 
      void run(rtc::TraceInterface &ti);
    };

#if RTC_DEVICE_CODE
    inline __device__ static 
    void run(rtc::TraceInterface &ti)
    {
      const int rayID
        = ti.getLaunchIndex().x
        + ti.getLaunchDims().x
        * ti.getLaunchIndex().y;
        
      auto &lp = OptixGlobals::get(ti);

      if (rayID >= lp.numRays)
        return;
        
      Ray &ray = lp.rays[rayID];
        
      vec3f dir = ray.dir;
      if (dir.x == 0.f) dir.x = 1e-6f;
      if (dir.y == 0.f) dir.y = 1e-6f;
      if (dir.z == 0.f) dir.z = 1e-6f;

      ti.traceRay(lp.world,
                  ray.org,
                  dir,
                  0.f,ray.tMax,
                  /* PRD */
                  (void *)&ray);
    }
#endif    
  }
  
      
  void Context::traceRaysLocally(GlobalModel *globalModel)
  {
    // ------------------------------------------------------------------
    // launch all in parallel ...
    // ------------------------------------------------------------------
    for (auto model : globalModel->modelSlots)
      for (auto device : *model->devices) {
        SetActiveGPU forDuration(device);
        render::OptixGlobals dd;
        auto ctx     = model->slotContext;
        dd.rays      = device->rayQueue->traceAndShadeReadQueue;
        dd.numRays   = device->rayQueue->numActive;
        dd.world     = model->getInstanceAccel(device);
        dd.materials = ctx->materialRegistry->getDD(device);
        dd.samplers  = ctx->samplerRegistry->getDD(device);
        dd.globalIndex = device->globalIndex;
        int bs = 1024;
        int nb = divRoundUp(dd.numRays,bs);
        device->traceRays->launch(vec2i(nb,bs),
                                  &dd);
      }
    
    // ------------------------------------------------------------------
    // ... and sync 'til all are done
    // ------------------------------------------------------------------
    syncCheckAll();
  }
  
  RTC_EXPORT_TRACE2D(traceRays,BARNEY_NS::render::TraceRays);
}

 
