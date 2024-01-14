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

#include "barney/umesh/UMeshField.h"
#include "barney/volume/MCAccelerator.h"
#include "cuBQL/bvh.h"

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



  template<typename TravState, int BVH_WIDTH=4>
  /*! sample the umesh field; can return NaN if sample did not hit
    any unstructured element at all */
  inline __device__
  float traverseCUQBL(typename cuBQL::WideBVH<float,3,BVH_WIDTH>::Node *bvhNodes,
                      TravState &ptd, vec3f P, bool dbg) 
  {
    printf("traversecuql\n");
    using node_t = typename cuBQL::WideBVH<float,3,BVH_WIDTH>::Node;
    struct NodeRef {
      union {
        struct {
          uint32_t offset:29;
          uint32_t count : 3;
        };
        uint32_t bits;
      };
    };
    NodeRef nodeRef;
    nodeRef.offset = 0;
    nodeRef.count  = 0;
    NodeRef stackBase[30];
    NodeRef *stackPtr = stackBase;
    while (true) {
      while (nodeRef.count == 0) {
        NodeRef childRef[BVH_WIDTH];
        node_t node = bvhNodes[nodeRef.offset];
#pragma unroll
        for (int i=0;i<BVH_WIDTH;i++) {
          const box3f &bounds = (const box3f&)node.children[i].bounds;
          if (node.children[i].valid && bounds.contains(P)) {
            childRef[i].offset = (int)node.children[i].offset;
            childRef[i].count  = (int)node.children[i].count;
          } else
            childRef[i].bits = 0;
        }
        nodeRef.bits = 0;
#pragma unroll
        for (int i=0;i<BVH_WIDTH;i++) {
          if (childRef[i].bits == 0)
            continue;
          if (nodeRef.bits == 0)
            nodeRef = childRef[i];
          else 
            *stackPtr++ = childRef[i];
        }
        if (nodeRef.bits == 0) {
          if (stackPtr == stackBase)
            return;
          nodeRef = *--stackPtr;
        }
      }
      // leaf ...
      if (ptd.leaf(P,nodeRef.offset,nodeRef.count) == false)
        return;
      // for (int i=0;i<nodeRef.count;i++) {
      //   auto elt = mesh.elements[nodeRef.offset+i];
      //   if (mesh.eltScalar(retVal,elt,P))
      //     return retVal;
      // }
      if (stackPtr == stackBase)
        return;
      nodeRef = *--stackPtr;
    }
  }
  
  struct UMeshSamplerPTD {
    inline __device__ UMeshSamplerPTD(const UMeshCUBQLSampler::DD *mesh)
      : mesh(mesh)
    {}
    inline __device__ bool leaf(vec3f P, int offset, int count)
    {
      printf("leaf ofs %i cnt %i\n",offset,count);
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
    if (!dbg) return NAN;
    
    UMeshSamplerPTD ptd(this);
    
    traverseCUQBL<UMeshSamplerPTD>(bvhNodes,ptd,P,dbg);
    return ptd.retVal;
    // // printf("bvh nodes %lx\n",bvhNodes);
    // return 0.f;
  }
  
}


