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
#include <cuBQL/bvh.h>

// #define AWT 1
#define AWT_MAX_DEPTH 6
  

namespace barney {

  struct __barney_align(16) AWTNode {
    enum { count_bits = 3, offset_bits = 32-count_bits, max_leaf_size = ((1<<count_bits)-1) };
    box4f   bounds[4];
    float   majorant[4];
    int     depth[4];
    struct NodeRef {
      uint32_t offset:offset_bits;
      uint32_t count :count_bits;
    };
    NodeRef child[4];
  };

  struct UMeshObjectSpace {
    struct DD {
      TransferFunction::DD xf;
      UMeshField::DD       mesh;
    };
  };
  
  /*! object-space accelerator that clusters elements into, well,
    clusters of similar/nearly elements, then builds an RTX BVH and
    majorants over those clusters, disables majorant-zero clusters
    during refit and in the isec program for a cluster perfomrs
    ray-element intersection followed by (per-element) woodock
    sampling along the ray-element overlap range */
  struct RTXObjectSpace : public VolumeAccel
  {
    struct Cluster {
      box4f bounds;
      int begin, end;
      float majorant;
    };
    
    struct DD : public UMeshObjectSpace::DD {
      Cluster             *clusters;
    };
    
    RTXObjectSpace(UMeshField *mesh, Volume *volume)
      : VolumeAccel(mesh,volume),
        mesh(mesh)
    {}
    static OWLGeomType createGeomType(DevGroup *devGroup);
    
    void build() override;

    std::vector<Cluster> clusters;
    OWLBuffer clustersBuffer = 0;
    void createClusters();

    OWLGeom  geom  = 0;
    OWLGroup group = 0;
    UMeshField *const mesh;
  };


  /*! object-space accelerator that clusters elements into, well,
    clusters of similar/nearly elements, then builds an RTX BVH and
    majorants over those clusters, disables majorant-zero clusters
    during refit and in the isec program for a cluster perfomrs
    ray-element intersection followed by (per-element) woodock
    sampling along the ray-element overlap range */
  struct UMeshAWT : public VolumeAccel
  {
    struct DD : public UMeshObjectSpace::DD {
      AWTNode             *nodes;
      int                 *roots;
    };
    
    UMeshAWT(UMeshField *mesh, Volume *volume)
      : VolumeAccel(mesh,volume),
        mesh(mesh)
    {}
    static OWLGeomType createGeomType(DevGroup *devGroup);
    
    void build() override;

    void buildNodes(cuBQL::WideBVH<float,3, 4> &qbvh);
    int extractRoots(cuBQL::WideBVH<float,3, 4> &qbvh,
                      int nodeID);
    void buildAWT();
    
    std::vector<int>     roots;
    std::vector<AWTNode> nodes;
    OWLBuffer nodesBuffer;
    OWLBuffer rootsBuffer;
    OWLGeom  geom  = 0;
    OWLGroup group = 0;
    UMeshField *const mesh;
  };
  

}
