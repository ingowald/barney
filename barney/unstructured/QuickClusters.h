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

#include "barney/unstructured/UMeshField.h"

namespace barney {

// #define CLUSTERS_FROM_QC 1
  
  struct Cluster {
    box4f bounds;
    float majorant;
#if CLUSTERS_FROM_QC
#else
    int begin, end;
#endif
  };

  struct UMeshQC : public UMeshField {
    enum { clusterSize = 8 };
    // enum { clusterSize = 16 };
  
    struct DD {
      UMeshField::DD       mesh;
      TransferFunction::DD xf;
      Cluster    *clusters;
    };
    
    enum { numHilbertBits = 20 };

    UMeshQC(DataGroup *owner,
             std::vector<vec4f> &vertices,
             std::vector<TetIndices> &tetIndices,
             std::vector<PyrIndices> &pyrIndices,
             std::vector<WedIndices> &wedIndices,
             std::vector<HexIndices> &hexIndices);
    uint64_t encodeHex(int primID);
    uint64_t encodeBox(const box4f &box4f);
    uint64_t encodeTet(int primID);
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "UMesh(via quick clusters){}"; }

    void build(Volume *volume) override;
    static OWLGeomType createGeomType(DevGroup *devGroup);

    box4f worldBounds;
    OWLGeom geom = 0;
  };

}
