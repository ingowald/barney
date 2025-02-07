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

/*! \file UMeshMC.dev.cu implements a macro-cell accelerated
    unstructured mesh data type.

    This particular voluem type:

    - uses cubql to accelerate point-in-element queries (for the
      scalar field evaluation)

    - uses macro cells and DDA traversal for domain traversal
*/

#include "barney/umesh/mc/UMeshCUBQLSampler.h"
#include "barney/volume/DDA.h"

RTC_DECLARE_GLOBALS(barney::render::OptixGlobals);

namespace barney {

  struct UMeshMC_Programs {
    
    template<typename TraceInterface>
    static inline __both__
    void bounds(const TraceInterface &ti,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    {
      MCVolumeAccel<UMeshCUBQLSampler>::boundsProg(ti,geomData,bounds,primID);
    }
    
    template<typename TraceInterface>
    static inline __both__
    void intersect(TraceInterface &ti)
    {
      MCVolumeAccel<UMeshCUBQLSampler>::isProg(ti);
    }
    
    template<typename TraceInterface>
    static inline __both__
    void closest_hit(TraceInterface &ti)
    { /* nothing to do */ }
    
    template<typename TraceInterface>
    static inline __both__
    void any_hit(TraceInterface &ti)
    { /* nothing to do */ }
  };

}

RTC_DECLARE_USER_GEOM(UMeshMC,barney::UMeshMC_Programs);


