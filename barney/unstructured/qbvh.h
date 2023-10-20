// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
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

#include "barney/common.h"

#define QBVH_DBG(a) /**/
#define QBVH_DBG_PRINT(a) QBVH_DBG(PRINT(a))
#define QBVH_DBG_PING QBVH_DBG(PING)

#define SKIP_TREE 1
#define QBVH_WIDTH 4

namespace qbvh {
  using namespace owl;
  using namespace owl::common;

  /*! a "child reference" - basically a means for a (multi-)node to
      reference what its child is (leaf node vs inner node), and
      "where it is" */
  struct ChildRef {
    inline __both__ bool valid() const { return bits != 0; }

    inline __both__ void makeLeaf(uint32_t index)
    {
      if (index >= (1<<30)) printf("very large index!?\n");
      ;
      bits = index | (1UL<<31);
    }
    inline __both__ void makeInner(uint32_t index) {
      if (index >= (1<<30)) printf("very large index!?\n");
      bits = index;
    }
    inline __both__ uint32_t getPrimIndex() const  { return bits & ~(1UL<<31); }
    inline __both__ uint32_t getChildIndex() const { return bits; }
    
    /*! whether this is a leaf node (true) or a inner node (false) */
    inline __both__ bool isLeaf() const { return (bits & (1UL<<31)); }
    uint32_t bits = 0;
  };
  
  struct Node {
    enum { numChildren = 4 };
    enum { NUM_CHILDREN = 4 };
    vec3f origin;
    float width;
    ChildRef childRef[QBVH_WIDTH];
    struct { 
      uint8_t lo[QBVH_WIDTH];
      uint8_t hi[QBVH_WIDTH];
    } dim[3];
    int32_t  skipTreeNode = -1;
    uint32_t skipTreeChild = 0;

    void makeInner(int slot, const box3f &bounds, int childNodeID)
    {
      for (int d=0;d<3;d++) {
        const float lower_bounds = origin[d];
        const float upper_bounds = origin[d]+width; 
        assert(lower_bounds <= bounds.lower[d]);
        assert(upper_bounds >= bounds.upper[d]);
        dim[d].lo[slot] = max(0,min(255,int(256.f*(bounds.lower[d]-lower_bounds)/width)));
        dim[d].hi[slot] = max(0,min(255,int(256.f*(upper_bounds-bounds.upper[d])/width)));
      }
      childRef[slot].makeInner((unsigned)childNodeID);
    }
    void makeLeaf(int slot, const BuildPrim &bp)
    {
      for (int d=0;d<3;d++) {
        const float lower_bounds = origin[d];
        const float upper_bounds = origin[d]+width;
        assert(lower_bounds <= bp.bounds.lower[d]);
        assert(upper_bounds >= bp.bounds.upper[d]);
        dim[d].lo[slot] = max(0,min(255,int(256.f*(bp.bounds.lower[d]-lower_bounds)/width)));
        dim[d].hi[slot] = max(0,min(255,int(256.f*(upper_bounds-bp.bounds.upper[d])/width)));
      }
      childRef[slot].makeLeaf((unsigned)bp.primID);
    }

    inline __both__ box3f getBounds(int slot) const
    {
      const vec3i lo_i(dim[0].lo[slot],
                       dim[1].lo[slot],
                       dim[2].lo[slot]);
      const vec3i hi_i(dim[0].hi[slot],
                       dim[1].hi[slot],
                       dim[2].hi[slot]);
      const box3f box((origin)         + vec3f(lo_i) * (width/256.f),
                      (origin + width) - vec3f(hi_i) * (width/256.f));
      return box;
    }
      
    void initQuantization(const box3f &bounds)
    {
      origin = bounds.lower;
      width  = reduce_max(bounds.span())*1.0001f;
    }
    void  clearAllAfter(int maxValid)
    {
      for (int slot=maxValid;slot<QBVH_WIDTH;slot++) {
        // invalid leaf: no inner node can point to root node:
        childRef[slot].makeInner(0);
        for (int d=0;d<3;d++) {
          dim[d].lo[slot] = 255;
          dim[d].hi[slot] =   0;
        }
      }
    }
  };

  struct BVH {
    box3f              worldBounds;
    std::vector<Node>  nodes;

    void setSkipNodes(int nodeID, int skipTreeNode, int skipTreeChild)
    {
      Node &node = nodes[nodeID];
      node.skipTreeNode = skipTreeNode;
      node.skipTreeChild = skipTreeChild;
      for (int i=0;i<QBVH_WIDTH;i++) {
        if (!node.childRef[i].valid()) continue;
        if (node.childRef[i].isLeaf()) continue;
          
        if (((i+1) < QBVH_WIDTH) && node.childRef[i+1].valid()) {
          // child HAS valid right brother
          setSkipNodes(node.childRef[i].getChildIndex(),
                       nodeID,i+1);
        } else {
          // no right brother for this child - skip node for this
          // child is same as for us
          setSkipNodes(node.childRef[i].getChildIndex(),
                       skipTreeNode,skipTreeChild);
        }
      }
    }
      
    void setSkipNodes()
    {
      setSkipNodes(0,-1,0);

      std::cout << "CHECKING skip tree values" << std::endl;
      for (int nodeID=0;nodeID<nodes.size();nodeID++) {
        auto &node = nodes[nodeID];
        if (node.skipTreeChild >= 4) {
          PING;
          PRINT(nodeID);
          PRINT(nodes.size());
          PRINT(node.skipTreeNode);
          PRINT(node.skipTreeChild);
        }
      }
      std::cout << "DONE checking skip tree values" << std::endl;
    }
        
  };

} // ::qbvh
