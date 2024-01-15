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

namespace barney {

  /*! a umesh scalar field, with a CUBQL bvh sampler */
  struct UMeshCUBQLSampler {
    enum { BVH_WIDTH = 4 };
    using bvh_t  = cuBQL::WideBVH<float,3,BVH_WIDTH>;
    using node_t = typename bvh_t::Node;
    
    struct DD : public UMeshField::DD {
      inline __device__ float sample(vec3f P, bool dbg = false) const;

      static void addVars(std::vector<OWLVarDecl> &vars, int base)
      {
        UMeshField::DD::addVars(vars,base);
        vars.push_back({"sampler.bvhNodes",OWL_BUFPTR,base+OWL_OFFSETOF(DD,bvhNodes)});
      }
  
      node_t  *bvhNodes;
    };

    struct Host {
      Host(ScalarField *sf) : mesh((UMeshField *)sf) {}

      /*! builds the string that allows for properly matching optix
          device progs for this type */
      inline std::string getTypeString() const { return "UMesh_CUBQL"; }

      void build(bool full_rebuild);

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
    inline __device__ UMeshSamplerPTD(const UMeshCUBQLSampler::DD *mesh)
      : mesh(mesh)
    {}
    inline __device__ bool leaf(vec3f P, int offset, int count)
    {
      for (int i=0;i<count;i++) {
        auto elt = mesh->elements[offset+i];
        if (mesh->eltScalar(retVal,elt,P))
          return false;
      }
      return true;
    }

    const UMeshCUBQLSampler::DD *const mesh;
    float retVal = NAN;
  };
  
  inline __device__
  float UMeshCUBQLSampler::DD::sample(vec3f P, bool dbg) const
  {
    UMeshSamplerPTD ptd(this);
    
    traverseCUQBL<UMeshSamplerPTD>(bvhNodes,ptd,P,dbg);
    return ptd.retVal;
  }
  
}


