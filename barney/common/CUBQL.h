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

#include "barney/common/barney-common.h"
#include "cuBQL/bvh.h"
#if BARNEY_CUBQL_HOST
# include "cuBQL/builder/host.h"
#else
# include "cuBQL/builder/cuda.h"
#endif

namespace barney {

  template<typename TravState, int BVH_WIDTH=4>
  /*! sample the umesh field; can return NaN if sample did not hit
    any unstructured element at all */
  inline __device__
  void traverseCUQBL(typename cuBQL::WideBVH<float,3,BVH_WIDTH>::Node *bvhNodes,
                      TravState &ptd, vec3f P, bool dbg) 
  {
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
      if (stackPtr == stackBase)
        return;
      nodeRef = *--stackPtr;
    }
  }




  template<typename TravState, int BVH_WIDTH=4>
  /*! sample the umesh field; can return NaN if sample did not hit
    any unstructured element at all */
  inline __device__
  void traverseCUQBL(typename cuBQL::BinaryBVH<float,3>::Node *bvhNodes,
                      TravState &ptd, vec3f P, bool dbg) 
  {
  }
} // ::barney
