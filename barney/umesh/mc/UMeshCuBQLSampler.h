// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/umesh/common/UMeshField.h"
#include "barney/common/CuBQL.h"
#include "cuBQL/traversal/fixedBoxQuery.h"

namespace BARNEY_NS {
  
  /*! a umesh scalar field, with a CuBQL bvh sampler */
  struct UMeshCuBQLSampler : public ScalarFieldSampler {
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
    
    UMeshCuBQLSampler(UMeshField *mesh);
    
    /*! builds the string that allows for properly matching optix
      device progs for this type */
    inline static std::string typeName() { return "UMesh_CuBQL"; }

    void build() override;

    UMeshField *const mesh;
    const DevGroup::SP devices;
  };
  
  inline __rtc_device
  float UMeshCuBQLSampler::DD::sample(vec3f P, bool dbg) const
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


