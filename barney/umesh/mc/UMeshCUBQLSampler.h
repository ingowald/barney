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
      inline __both__ float sample(vec3f P, bool dbg = false) const;
      
      node_t  *bvhNodes;
    };
    DD getDD(Device *device);
    
    /*! per-device data - parent store the umesh, we just store the
      bvh nodes */
    struct PLD {
      node_t *bvhNodes = 0;
      // bvh_t bvh;
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;
    
    /*! for-cubql traversal state that we can use with a cubql
      traversal call back */
    struct Traversal {
      inline __both__ Traversal(const UMeshCUBQLSampler::DD *const mesh, bool dbg);
      inline __both__ bool leaf(vec3f P, int offset, int count);
      
      const UMeshCUBQLSampler::DD *const mesh;
      float retVal = NAN;
      bool const dbg;
    };
    
    UMeshCUBQLSampler(UMeshField *mesh);
    
    /*! builds the string that allows for properly matching optix
      device progs for this type */
    inline static std::string typeName() { return "UMesh_CUBQL"; }

    void build() override;

    UMeshField *const mesh;
    const DevGroup::SP devices;
  };
  
  inline __both__
  bool UMeshCUBQLSampler::Traversal::leaf(vec3f P, int offset, int count)
  {
    // if (dbg) printf("at leaf %i %i\n",offset,count);
    for (int i=0;i<count;i++) {
      auto elt = mesh->elements[offset+i];
      // if (dbg) printf("elt type %i id %i, mesh %p\n",
      //                 elt.type,elt.ofs0,mesh);
      if (mesh->eltScalar(retVal,elt,P,dbg))
        return false;
    }
    return true;
  }

  inline __both__
  UMeshCUBQLSampler::Traversal::Traversal(const UMeshCUBQLSampler::DD *const mesh,
                                          bool dbg)
    : mesh(mesh), dbg(dbg)
  {}
  
  inline __both__
  float UMeshCUBQLSampler::DD::sample(vec3f P, bool dbg) const
  {
    UMeshCUBQLSampler::Traversal traversal(this,dbg);
    typename bvh_t::box_t box; box.lower = box.upper = to_cubql(P);
    bvh_t bvh;
    bvh.nodes = bvhNodes;
    bvh.primIDs = nullptr;
    
    auto lambda = [&](const uint32_t *primIDs, int numPrims)
    {
      if (traversal.leaf(P,int(primIDs - bvh.primIDs), numPrims))
        return CUBQL_CONTINUE_TRAVERSAL;
      else
        return CUBQL_TERMINATE_TRAVERSAL;
    };
    cuBQL::fixedBoxQuery::forEachLeaf(lambda,bvh,box);
    return traversal.retVal;
  }
  
} // ::barney


