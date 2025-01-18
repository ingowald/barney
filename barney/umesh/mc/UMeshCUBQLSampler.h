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

#include "barney/umesh/common/UMeshField.h"
#include "barney/volume/MCAccelerator.h"
#include "barney/common/CUBQL.h"
#include "cuBQL/traversal/fixedBoxQuery.h"

namespace barney {

  /*! a umesh scalar field, with a CUBQL bvh sampler */
  struct UMeshCUBQLSampler {
#if 1
    using bvh_t  = cuBQL::BinaryBVH<float,3>;
#else
    enum { BVH_WIDTH = 4 };
    using bvh_t  = cuBQL::WideBVH<float,3,BVH_WIDTH>;
#endif
    using node_t = typename bvh_t::Node;
    
    struct DD : public UMeshField::DD {
      inline __both__ float sample(vec3f P, bool dbg = false) const;

      // static void addVars(std::vector<OWLVarDecl> &vars, int base)
      // {
      //   UMeshField::DD::addVars(vars,base);
      //   vars.push_back({"sampler.bvhNodes",OWL_BUFPTR,base+OWL_OFFSETOF(DD,bvhNodes)});
      // }
  
      node_t  *bvhNodes;
    };

    struct Host {
      Host(ScalarField *sf) : mesh((UMeshField *)sf) {}

      /*! builds the string that allows for properly matching optix
          device progs for this type */
      inline std::string getTypeString() const { return "UMesh_CUBQL"; }

      void build(bool full_rebuild);

      const std::vector<Device::SP> &getDevices();
      DevGroup *getDevGroup();
      
      void setVariables(OWLGeom geom)
      {
        owlGeomSetBuffer(geom,"sampler.bvhNodes",bvhNodesBuffer);
      }
      
      OWLBuffer   bvhNodesBuffer = 0;
      UMeshField *const mesh;
    };
  };


  /*! per-traversal data for the cuBQLTrave callback */
  struct UMeshSamplerPTD {
    inline __device__ UMeshSamplerPTD(const UMeshCUBQLSampler::DD *mesh, bool dbg)
      : mesh(mesh), dbg(dbg)
    {}
    inline __device__ bool leaf(vec3f P, int offset, int count)
    {
      // if (dbg) printf("cubql leaf P %f %f %f ofs %i cnt %i\n",
      //                 P.x,P.y,P.z,offset,count);
      for (int i=0;i<count;i++) {
        auto elt = mesh->elements[offset+i];
        if (mesh->eltScalar(retVal,elt,P,dbg))
          return false;
      }
      return true;
    }

    const UMeshCUBQLSampler::DD *const mesh;
    float retVal = NAN;
    bool const dbg;
  };
  
  inline __device__
  float UMeshCUBQLSampler::DD::sample(vec3f P, bool dbg) const
  {
    UMeshSamplerPTD ptd(this,dbg);
    //    traverseCUQBL<UMeshSamplerPTD>(bvhNodes,ptd,P,dbg);
    typename bvh_t::box_t box; box.lower = box.upper = P;
    bvh_t bvh;
    bvh.nodes = bvhNodes;
    bvh.primIDs = nullptr;

    auto lambda = [&](const uint32_t *primIDs, int numPrims)
    {
      if (ptd.leaf(P,primIDs - bvh.primIDs, numPrims))
        return CUBQL_CONTINUE_TRAVERSAL;
      else
        return CUBQL_TERMINATE_TRAVERSAL;
    };
    cuBQL::fixedBoxQuery::forEachLeaf(lambda,bvh,box);
    return ptd.retVal;
  }
  
}


