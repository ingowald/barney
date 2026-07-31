// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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
      virtual ~Host() {}
    // RTXObjectSpace(UMeshField *mesh, Volume *volume)
    //   : VolumeAccel(mesh,volume),
    //     mesh(mesh)
    // {}
      static OWLGeomType createGeomType(DevGroup *devGroup);
    
      void build(bool full_rebuild) override;
      void createClusters();

      /*! set owl variables for this accelerator - this is virutal so
        derived classes can add their own */
      // void setVariables(OWLGeom geom) override;
      
      std::vector<Cluster> clusters;
      OWLBuffer clustersBuffer = 0;
      bool firstTimeBuild = true;
      // OWLGeom  geom  = 0;
      // OWLGroup group = 0;
      // UMeshField *const mesh;
    };
  };
  
}

