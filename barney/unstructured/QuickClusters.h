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

  inline __both__
  vec3f getPos(vec4f v)
  { return {v.x,v.y,v.z}; }

  inline __both__
  box3f getBox(box4f bb)
  { return {getPos(bb.lower),getPos(bb.upper)}; }

  inline __both__
  range1f getRange(box4f bb)
  { return {bb.lower.w,bb.upper.w}; }

  
  struct DeviceXF {
    inline __device__ vec4f map(float s) const;
    inline __device__ float majorant(range1f r) const;

    float4  *values;
    range1f  domain;
    float    baseDensity;
    int      numValues;
  };

  struct Cluster {
    box4f bounds;
    float majorant;
  };

  struct UMeshQC : public UMeshField {
    // enum { clusterSize = 1 };
    enum { clusterSize = 16 };
    struct Element {
      uint32_t ID:29;
      uint32_t type:3;
    };
  
    struct DD {
      inline __device__
      box4f getBounds(Element element) const;

      // inline __device__
      // bool sampleAndMap(vec4f &color,
      //                   int elts_begin, int elts_end,
      //                   vec3f position) const;
                        
      // inline __device__
      // bool sample(float &scalar,
      //             Element element,
      //             vec3f position) const;
      // inline __device__
      // bool sampleAndMap(vec4f &color,
      //                   Element element,
      //                   vec3f position) const;
      
      DeviceXF xf;

      float4     *vertices;
      int4       *tetIndices;
      HexIndices *hexIndices;
      Element    *elements;
      int         numElements;
      Cluster    *clusters;
    };
    
    enum { numHilbertBits = 20 };
    enum { TET=0, HEX } ElementType;

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
  };

  inline __device__
  box4f UMeshQC::DD::getBounds(Element element) const
  {
    switch (element.type) {
    case UMeshQC::TET: {
      const int *indices = (const int *)&tetIndices[element.ID];
      return box4f()
        .including(vertices[indices[0]])
        .including(vertices[indices[1]])
        .including(vertices[indices[2]])
        .including(vertices[indices[3]]);
    }
    default:
      return box4f(); 
    }
  }

}
