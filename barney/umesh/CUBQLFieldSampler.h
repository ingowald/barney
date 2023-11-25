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

#include "barney/volume/Volume.h"
#include "barney/umesh/UMeshField.h"
#include <cuBQL/bvh.h>

namespace barney {

   /*! can sample umeshes using cuda-point location of cuBQL-bvh. this
       samples a *field* and thus doesn't know anything about transfer
       functions or acceleration */
  struct CUBQLFieldSampler {
    enum { BVH_WIDTH = 4 };
    using bvh_t  = cuBQL::WideBVH<float,3,BVH_WIDTH>;
    using node_t = typename bvh_t::Node;
    
    struct DD {
      /*! sample the umesh field; can return NaN if sample did not hit
        any unstructured element at all */
      inline __device__ float sample(vec3f P, bool dbg=false) const;

      node_t         *bvhNodes;
      UMeshField::DD  mesh;
    };
    
    CUBQLFieldSampler(ScalarField *field);
    void build();

    UMeshField *const mesh;
    OWLBuffer   bvhNodesBuffer = 0;
  };

  /*! sample the umesh field; can return NaN if sample did not hit
    any unstructured element at all */
  inline __device__
  float CUBQLFieldSampler::DD::sample(vec3f P, bool dbg) const
  {
    float retVal = NAN;
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
            return retVal;
          nodeRef = *--stackPtr;
        }
      }
      // leaf ...
      for (int i=0;i<nodeRef.count;i++) {
        auto elt = mesh.elements[nodeRef.offset+i];
        if (mesh.eltScalar(retVal,elt,P))
          return retVal;
      }
      if (stackPtr == stackBase)
        return retVal;
      nodeRef = *--stackPtr;
    }
    return retVal;
  }
  
}
