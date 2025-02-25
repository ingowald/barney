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

#include "barney/amr/BlockStructuredField.h"
#include "barney/volume/MCAccelerator.h"
#include "barney/common/CUBQL.h"
#include "cuBQL/traversal/fixedBoxQuery.h"

namespace BARNEY_NS {

#if 0
  /*! a umesh scalar field, with a CUBQL bvh sampler */
  struct BlockStructuredCUBQLSampler {
    enum { BVH_WIDTH = 4 };
    using bvh_t  = cuBQL::WideBVH<float,3,BVH_WIDTH>;
    using node_t = typename bvh_t::Node;
    
    struct DD : public BlockStructuredField::DD {
      inline __device__ float sample(vec3f P, bool dbg = false) const;

      // static void addVars(std::vector<OWLVarDecl> &vars, int base)
      // {
      //   BlockStructuredField::DD::addVars(vars,base);
      //   vars.push_back({"sampler.bvhNodes",OWL_BUFPTR,base+OWL_OFFSETOF(DD,bvhNodes)});
      // }
  
      node_t  *bvhNodes;
    };

    struct Host {
      Host(ScalarField *sf) : field((BlockStructuredField *)sf) {}

      /*! builds the string that allows for properly matching optix
          device progs for this type */
      inline std::string getTypeString() const { return "BlockStructured"; }

      void build(bool full_rebuild);

      void setVariables(OWLGeom geom)
      {
        owlGeomSetBuffer(geom,"sampler.bvhNodes",bvhNodesBuffer);
      }
      
      OWLBuffer   bvhNodesBuffer = 0;
      BlockStructuredField *const field;
    };
  };

  struct BlockStructuredSamplerPTD {
    inline __device__ BlockStructuredSamplerPTD(const BlockStructuredCUBQLSampler::DD *field)
      : field(field)
    {}
    inline __device__ bool leaf(vec3f P, int offset, int count)
    {
      for (int i=0;i<count;i++) {
        auto blockID = field->blockIDs[offset+i];
        field->addBasisFunctions(sumWeightedValues,sumWeights,blockID,P);
      }
      return true;
    }

    const BlockStructuredCUBQLSampler::DD *const field;
    float sumWeights = 0.f;
    float sumWeightedValues = 0.f;
  };
  
  inline __device__
  float BlockStructuredCUBQLSampler::DD::sample(vec3f P, bool dbg) const
  {
    BlockStructuredSamplerPTD ptd(this);

    bvh_t bvh;
    bvh.nodes = bvhNodes;
    bvh.primIDs = nullptr;
    auto lambda = [&](const uint32_t *primIDs, int numPrimsInLeaf) -> int {
      int offset = primIDs - bvh.primIDs;
      ptd.leaf(P,offset,numPrimsInLeaf);
      return CUBQL_CONTINUE_TRAVERSAL;
    };
    cuBQL::box3f box; box.lower = box.upper = P;
    cuBQL::fixedBoxQuery::forEachLeaf(lambda,bvh,box);
    // traverseCUQBL<BlockStructuredSamplerPTD>(bvhNodes,ptd,P,dbg);
    return ptd.sumWeights == 0.f ? NAN : (ptd.sumWeightedValues  / ptd.sumWeights);
  }
#endif
  
}


