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

#include "barney/Volume.h"
#include "barney/unstructured/UMeshField.h"

namespace barney {

  /*! object-space accelerator that clusters elements into, well,
    clusters of similar/nearly elements, then builds an RTX BVH and
    majorants over those clusters, disables majorant-zero clusters
    during refit and in the isec program for a cluster perfomrs
    ray-element intersection followed by (per-element) woodock
    sampling along the ray-element overlap range */
  struct UMeshRTXObjectSpace : public VolumeAccel
  {
    struct Cluster {
      box4f bounds;
      int begin, end;
      float majorant;
    };
    
    struct DD {
      TransferFunction::DD xf;
      UMeshField::DD       mesh;
      Cluster             *clusters;
    };
    
    UMeshRTXObjectSpace(UMeshField *mesh, Volume *volume)
      : VolumeAccel(mesh,volume),
        mesh(mesh)
    {}
    static OWLGeomType createGeomType(DevGroup *devGroup);
    
    void build() override;
    void createClusters();
    
    std::vector<Cluster> clusters;
    OWLBuffer clustersBuffer = 0;
    OWLGeom  geom  = 0;
    OWLGroup group = 0;
    UMeshField *const mesh;
  };

}
