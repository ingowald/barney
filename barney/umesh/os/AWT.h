// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/umesh/common/UMeshField.h"
#include "barney/volume/Volume.h"

// #define AWT_DEFAULT_MAX_DEPTH 7
#define AWT_NODE_WIDTH 4

namespace BARNEY_NS {

  struct RefitInfo {
    int numNotDone;
    int parent;
  };
  
  struct __barney_align(16) AWTNode {
    enum { count_bits = 3,
           offset_bits = 32-count_bits,
           max_leaf_size = ((1<<count_bits)-1) };
    // int     depth[4];
    struct NodeRef {
      inline __rtc_device bool valid() const { return count != 0 || offset != 0; }
      inline __rtc_device bool isLeaf() const { return count != 0; }
      uint32_t offset:offset_bits;
      uint32_t count :count_bits;
    };
    struct __barney_align(16) Child {
      box3f    bounds;
      float    majorant;
      NodeRef  nodeRef;
    };
    Child child[AWT_NODE_WIDTH];
  };


  struct AWTAccel : public VolumeAccel  {
    struct DD
    {
      // box3f                 bounds;
      UMeshField::DD        mesh;
      TransferFunction::DD  xf;
      AWTNode              *awtNodes;
      uint32_t             *primIDs;
      int                   userID;
    };

    struct PLD {
      AWTNode      *awtNodes   = 0;
      uint32_t     *primIDs    = 0;
      RefitInfo    *refitInfos = 0;
      rtc::Geom    *geom       = 0;
      rtc::Group   *group      = 0;
      rtc::ComputeKernel1D *copyNodes  = 0;
      rtc::ComputeKernel1D *computeMajorants = 0;
      int           numNodes   = 0;
      box3f         bounds;
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;

    DD getDD(Device *device);

    AWTAccel(Volume *volume,
             UMeshField *mesh);
    
    void build(bool full_rebuild) override;

    UMeshField *const mesh;
  };
    
}
