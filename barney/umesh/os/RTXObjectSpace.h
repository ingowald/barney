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

namespace barney {

  /*! object-space accelerator that clusters elements into, well,
    clusters of similar/nearly elements, then builds an RTX BVH and
    majorants over those clusters, disables majorant-zero clusters
    during refit and in the isec program for a cluster perfomrs
    ray-element intersection followed by (per-element) woodock
    sampling along the ray-element overlap range */
  struct RTXObjectSpace
  {
    struct Cluster {
      box4f bounds;
      int begin, end;
      float majorant;
    };
    
    struct DD : public UMeshObjectSpace::DD {
      using Inherited = UMeshObjectSpace::DD;
      static void addVars(std::vector<OWLVarDecl> &vars, int base);
      
      Cluster             *clusters;
      /*! true (only) if this is the first time this is being built */
      int                  firstTimeBuild;
    };
    
    struct Host : public UMeshObjectSpace::Host {
      using Inherited = UMeshObjectSpace::Host;
      Host(UMeshField *mesh, Volume *volume)
        : Inherited(mesh,volume)
      {}
    // RTXObjectSpace(UMeshField *mesh, Volume *volume)
    //   : VolumeAccel(mesh,volume),
    //     mesh(mesh)
    // {}
      static OWLGeomType createGeomType(DevGroup *devGroup);
    
      void build(bool full_rebuild) override;
      void createClusters();

      /*! set owl variables for this accelerator - this is virutal so
        derived classes can add their own */
      void setVariables(OWLGeom geom) override;
      
      std::vector<Cluster> clusters;
      OWLBuffer clustersBuffer = 0;
      bool firstTimeBuild = true;
      // OWLGeom  geom  = 0;
      // OWLGroup group = 0;
      // UMeshField *const mesh;
    };
  };
  
}

