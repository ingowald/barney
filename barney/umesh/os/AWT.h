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

#include "barney/umesh/os/ObjectSpace-common.h"

#define AWT_DEFAULT_MAX_DEPTH 7

namespace barney {

  struct __barney_align(16) AWTNode {
    enum { count_bits = 3, offset_bits = 32-count_bits, max_leaf_size = ((1<<count_bits)-1) };
    box4f   bounds[4];
    float   majorant[4];
    // int     depth[4];
    struct NodeRef {
      inline __both__ bool valid() const { return count != 0 || offset != 0; }
      inline __both__ bool isLeaf() const { return count != 0; }
      uint32_t offset:offset_bits;
      uint32_t count :count_bits;
    };
    NodeRef child[4];
  };

  /*! object-space accelerator that clusters elements into, well,
    clusters of similar/nearly elements, then builds an RTX BVH and
    majorants over those clusters, disables majorant-zero clusters
    during refit and in the isec program for a cluster perfomrs
    ray-element intersection followed by (per-element) woodock
    sampling along the ray-element overlap range */
  struct UMeshAWT
  {
    struct DD : public UMeshObjectSpace::DD {
      using Inherited = UMeshObjectSpace::DD;
      static void addVars(std::vector<OWLVarDecl> &vars, int base);
      
      AWTNode             *nodes;
      int                 *roots;
    };

    struct Host : public UMeshObjectSpace::Host {
      using Inherited = UMeshObjectSpace::Host;
      Host(UMeshField *mesh, Volume *volume)
        : Inherited(mesh,volume)
      {}
      static OWLGeomType createGeomType(DevGroup *devGroup);
      
      void build(bool full_rebuild) override;
      
      void buildNodes(cuBQL::WideBVH<float,3, 4> &qbvh);
      void extractRoots();
      void buildAWT();
      
      std::vector<int>     roots;
      std::vector<AWTNode> nodes;
      OWLBuffer nodesBuffer;
      OWLBuffer rootsBuffer;
    };
  };
  
}
