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

#pragma once

#include "barney/material/DeviceMaterial.h"
#include "barney/render/Sampler.h"
#include "barney/render/HitAttributes.h"
#if BARNEY_DEVICE_PROGRAM
# include "rtcore/TraceInterface.h"
#endif
#include "barney/render/World.h"
#include "barney/render/HitIDs.h"

namespace BARNEY_NS {
  namespace render {

    /*! defines all constant global launch parameter data. The struct
        type itself is also defined on the host (so its contents can
        be marshalled there, but the 'get()' method can only be
        available in device programs */
    struct OptixGlobals {
#if BARNEY_DEVICE_PROGRAM
      static inline __rtc_device
      const OptixGlobals &get(const rtc::TraceInterface &dev);
#endif

      /*! the current ray queue for the traceRays() kernel */
      Ray                   *rays;
      /*! info for primid/geomid/instid info; may be null if not required */
      HitIDs                *hitIDs;
      
      /*! number of ryas in the queue */
      int                    numRays;
      
      /*! this device's world to trace rays into */
      rtc::device::AccelHandle  accel;
      World::DD                 world;
    };
  }
}

namespace BARNEY_NS {
  namespace render {

#if BARNEY_DEVICE_PROGRAM
    inline __rtc_device
    const OptixGlobals &OptixGlobals::get(const rtc::TraceInterface &be)
    {
      return *(OptixGlobals*)be.getLPData();
    }
#endif
  }
}
