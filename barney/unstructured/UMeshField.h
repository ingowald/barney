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
#include "barney/DataGroup.h"
#include "barney/unstructured/MCGrid.h"

namespace barney {

  /*! helper functoin to extrace 3f spatial component from 4f point-plus-scalar */
  inline __both__ vec3f getPos(vec4f v)
  {return vec3f{v.x,v.y,v.z}; }

  /*! helper functoin to extrace 3f spatial component from 4f point-plus-scalar */
  inline __both__ box3f getBox(box4f bb)
  { return box3f{getPos(bb.lower),getPos(bb.upper)}; }

  /*! helper functoin to extract 1f scalar range from 4f point-plus-scalar */
  inline __both__ range1f getRange(box4f bb)
  { return range1f{bb.lower.w,bb.upper.w}; }

  
  struct Element {
    typedef enum { TET=0, HEX } Type;
    
    inline __both__ Element() {}
    inline __both__ Element(int ID, int type)
      : ID(ID), type(type)
    {}
    uint32_t ID:29;
    uint32_t type:3;
  };
  
  struct UMeshField : public ScalarField {
    
    typedef std::shared_ptr<UMeshField> SP;

    template<int N>
    struct ints { int v[N];
      inline __both__ int &operator[](int i)      { return v[i]; }
      inline __both__ int operator[](int i) const { return v[i]; }
    };
    struct DD {
      inline __both__ box4f elementBounds(Element element) const;
      inline __both__ box4f tetBounds(int primID) const;
      inline __both__ box4f hexBounds(int primID) const;
      
      const float4     *vertices;
      const int4       *tetIndices;
      const ints<8>    *hexIndices;
      const Element    *elements;
      int               numElements;
    };

    /*! build *initial* macro-cell grid (ie, the scalar field min/max
        ranges, but not yet the majorants) over a umesh */
    void buildInitialMacroCells(MCGrid &grid);

    UMeshField(DevGroup *devGroup,
               std::vector<vec4f> &vertices,
               std::vector<TetIndices> &tetIndices,
               std::vector<PyrIndices> &pyrIndices,
               std::vector<WedIndices> &wedIndices,
               std::vector<HexIndices> &hexIndices);

    DD getDD(int devID);
    
    VolumeAccel::SP createAccel(Volume *volume) override;
    void buildParams(std::vector<OWLVarDecl> &params, size_t offset);
    void setParams(OWLLaunchParams lp);

    std::vector<vec4f>      vertices;
    std::vector<TetIndices> tetIndices;
    std::vector<PyrIndices> pyrIndices;
    std::vector<WedIndices> wedIndices;
    std::vector<HexIndices> hexIndices;
    std::vector<Element>    elements;
    
    OWLBuffer verticesBuffer   = 0;
    OWLBuffer tetIndicesBuffer = 0;
    OWLBuffer hexIndicesBuffer = 0;
    OWLBuffer elementsBuffer   = 0;

    box4f worldBounds;
  };


  /*! computes - ON CURRENT DEVICE - the given mesh's prim bounds, and
      writes those into givne pre-allocated device mem location */
  __global__
  void computeElementBoundingBoxes(box3f *d_primBounds, UMeshField::DD mesh);
  
  // ==================================================================
  // IMPLEMENTATION
  // ==================================================================
  
  inline __both__
  box4f UMeshField::DD::tetBounds(int tetID) const
  {
    const int4 indices = tetIndices[tetID];
    return box4f()
      .including(vertices[indices.x])
      .including(vertices[indices.y])
      .including(vertices[indices.z])
      .including(vertices[indices.w]);
  }
  
  inline __both__
  box4f UMeshField::DD::hexBounds(int hexID) const
  {
    UMeshField::ints<8> indices = hexIndices[hexID];
    return box4f()
      .including(vertices[indices[0]])
      .including(vertices[indices[1]])
      .including(vertices[indices[2]])
      .including(vertices[indices[3]])
      .including(vertices[indices[4]])
      .including(vertices[indices[5]])
      .including(vertices[indices[6]])
      .including(vertices[indices[7]]);
  }
  
  inline __both__
  box4f UMeshField::DD::elementBounds(Element element) const
  {
    switch (element.type) {
    case Element::TET: 
      return tetBounds(element.ID);
    case Element::HEX: 
      return hexBounds(element.ID);
    }
    // ugh: could not recognize this element type!?
    return box4f(); 
  }

}
