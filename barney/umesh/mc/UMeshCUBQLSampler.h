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

#pragma once

#include "barney/umesh/common/UMeshField.h"
#include "barney/common/CUBQL.h"
#include "cuBQL/traversal/fixedBoxQuery.h"

namespace BARNEY_NS {
  
  /*! a umesh scalar field, with a CUBQL bvh sampler */
  struct UMeshCUBQLSampler : public ScalarFieldSampler {
#if 1
    using bvh_t  = cuBQL::BinaryBVH<float,3>;
#else
    enum { BVH_WIDTH = 4 };
    using bvh_t  = cuBQL::WideBVH<float,3,BVH_WIDTH>;
#endif
    using node_t = typename bvh_t::Node;
    
    struct DD : public UMeshField::DD {
      inline __rtc_device float sample(vec3f P, bool dbg = false) const;
      
      bvh_t bvh;
    };
    DD getDD(Device *device);
    
    /*! per-device data - parent store the umesh, we just store the
      bvh nodes */
    struct PLD {
      bvh_t bvh = { 0,0,0,0 };
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;
    
    UMeshCUBQLSampler(UMeshField *mesh);
    
    /*! builds the string that allows for properly matching optix
      device progs for this type */
    inline static std::string typeName() { return "UMesh_CUBQL"; }

    void build() override;

    UMeshField *const mesh;
    const DevGroup::SP devices;
  };
  
  inline __rtc_device
  float UMeshCUBQLSampler::DD::sample(vec3f P, bool dbg) const
  {
    typename bvh_t::box_t box; box.lower = box.upper = to_cubql(P);

    float retVal = NAN;
    auto lambda = [this,P,&retVal,dbg]
      (const uint32_t primID)
    {
      if (this->eltScalar(retVal,primID,P,dbg))
        return CUBQL_TERMINATE_TRAVERSAL;
      return CUBQL_CONTINUE_TRAVERSAL;
    };
    cuBQL::fixedBoxQuery::forEachPrim(lambda,bvh,box);
    return retVal;
  }
  
} // ::BARNEY_NS


