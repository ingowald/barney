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

#include "barney/DataGroup.h"
// #include "barney/volume/MCAccelerator.h"
/* all routines for point-element sampling/intersection - shold
   logically be part of this file, but kept in separate file because
   these were mostly imported from oepnvkl */
#include "barney/umesh/common/ElementIntersection.h"

namespace barney {

  struct Element {
    typedef enum { TET=0, PYR, WED, HEX, GRID } Type;
    
    inline __both__ Element() {}
    inline __both__ Element(int ID, int type)
      : ID(ID), type(type)
    {}
    uint32_t ID  :29;
    uint32_t type: 3;
  };

  struct UMeshField : public ScalarField
  {
    typedef std::shared_ptr<UMeshField> SP;

    /*! helper class for representing an N-long integer tuple, to
       represent wedge, pyramid, hex, etc elemnet indices */
    template<int N>
    struct ints { int v[N];
      inline __both__ int &operator[](int i)      { return v[i]; }
      inline __both__ int operator[](int i) const { return v[i]; }
    };
    /*! device-data for a unstructured-mesh scalar field, containing
        all device-side pointers and function to access this field and
        sample/evaluate its elemnets */
    struct DD : public ScalarField::DD {
      static void addVars(std::vector<OWLVarDecl> &vars, int base);
      
      inline __both__ box4f eltBounds(Element element) const;
      inline __both__ box4f tetBounds(int primID) const;
      inline __both__ box4f pyrBounds(int primID) const;
      inline __both__ box4f wedBounds(int primID) const;
      inline __both__ box4f hexBounds(int primID) const;
      inline __both__ box4f gridBounds(int primID) const;

      /* compute scalar of given umesh element at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __both__ bool eltScalar(float &retVal, Element elt, vec3f P) const;
      
      /* compute scalar of given tet in umesh, at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __both__ bool tetScalar(float &retVal, int primID, vec3f P) const;
      
      /* compute scalar of given pyramid in umesh, at point P, and
         return that in 'retVal'. returns true if P is inside the
         elemnt, false if outside (in which case retVal is not
         defined) */
      inline __both__ bool pyrScalar(float &retVal, int primID, vec3f P) const;
      
      /* compute scalar of given wedge in umesh, at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __both__ bool wedScalar(float &retVal, int primID, vec3f P) const;
      
      /* compute scalar of given hex in umesh, at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __both__ bool hexScalar(float &retVal, int primID, vec3f P) const;
      
      /* compute scalar of given grid in umesh, at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __both__ bool gridScalar(float &retVal, int primID, vec3f P) const;
      
      const float4     *vertices;
      const int4       *tetIndices;
      const ints<5>    *pyrIndices;
      const ints<6>    *wedIndices;
      const ints<8>    *hexIndices;
      const Element    *elements;
      const int        *gridOffsets;
      const vec3i      *gridDims;
      const box4f      *gridDomains;
      const float      *gridScalars;
      int               numElements;
    };

    // std::vector<OWLVarDecl> getVarDecls(uint32_t myOfs) override;
    void setVariables(OWLGeom geom) override;
    
    /*! build *initial* macro-cell grid (ie, the scalar field min/max
      ranges, but not yet the majorants) over a umesh */
    void buildInitialMacroCells(MCGrid &grid);

    void buildMCs(MCGrid &macroCells) override;
    
    /*! computes, on specified device, the bounding boxes and - if
      d_primRanges is non-null - the primitmives ranges. d_primBounds
      and d_primRanges (if non-null) must be pre-allocated and
      writeaable on specified device */
    void computeElementBBs(int deviceID,
                           box3f *d_primBounds,
                           range1f *d_primRanges=0);
    
    UMeshField(DataGroup *owner,
               std::vector<vec4f> &vertices,
               std::vector<TetIndices> &tetIndices,
               std::vector<PyrIndices> &pyrIndices,
               std::vector<WedIndices> &wedIndices,
               std::vector<HexIndices> &hexIndices,
               std::vector<int> &gridOffsets,
               std::vector<vec3i> &gridDims,
               std::vector<box4f> &gridDomains,
               std::vector<float> &gridScalars,
               const box3f &domain);

    DD getDD(int devID);
    
    VolumeAccel::SP createAccel(Volume *volume) override;

    /*! returns part of the string used to find the optix device
        programs that operate on this type */
    std::string getTypeString() const { return "UMesh"; };
    
    std::vector<vec4f>      vertices;
    std::vector<TetIndices> tetIndices;
    std::vector<PyrIndices> pyrIndices;
    std::vector<WedIndices> wedIndices;
    std::vector<HexIndices> hexIndices;
    std::vector<Element>    elements;
    std::vector<int>        gridOffsets;
    std::vector<vec3i>      gridDims;
    std::vector<box4f>      gridDomains;
    std::vector<float>      gridScalars;
    
    OWLBuffer verticesBuffer   = 0;
    OWLBuffer tetIndicesBuffer = 0;
    OWLBuffer pyrIndicesBuffer = 0;//todo wire in
    OWLBuffer wedIndicesBuffer = 0;//todo wire in
    OWLBuffer hexIndicesBuffer = 0;
    OWLBuffer elementsBuffer   = 0;
    OWLBuffer gridOffsetsBuffer = 0;
    OWLBuffer gridDimsBuffer = 0;
    OWLBuffer gridDomainsBuffer = 0;
    OWLBuffer gridScalarsBuffer = 0;
  };
  
  // ==================================================================
  // IMPLEMENTATION
  // ==================================================================
  
  inline __both__
  box4f UMeshField::DD::tetBounds(int tetID) const
  {
    const int4 indices = tetIndices[tetID];
    return box4f()
      .including(make_vec4f(vertices[indices.x]))
      .including(make_vec4f(vertices[indices.y]))
      .including(make_vec4f(vertices[indices.z]))
      .including(make_vec4f(vertices[indices.w]));
  }

  inline __both__
  box4f UMeshField::DD::pyrBounds(int pyrID) const
  {
    UMeshField::ints<5> indices = pyrIndices[pyrID];
    return box4f()
      .including(make_vec4f(vertices[indices[0]]))
      .including(make_vec4f(vertices[indices[1]]))
      .including(make_vec4f(vertices[indices[2]]))
      .including(make_vec4f(vertices[indices[3]]))
      .including(make_vec4f(vertices[indices[4]]));
  }

  inline __both__
  box4f UMeshField::DD::wedBounds(int wedID) const
  {
    UMeshField::ints<6> indices = wedIndices[wedID];
    return box4f()
      .including(make_vec4f(vertices[indices[0]]))
      .including(make_vec4f(vertices[indices[1]]))
      .including(make_vec4f(vertices[indices[2]]))
      .including(make_vec4f(vertices[indices[3]]))
      .including(make_vec4f(vertices[indices[4]]))
      .including(make_vec4f(vertices[indices[5]]));
  }
  
  inline __both__
  box4f UMeshField::DD::hexBounds(int hexID) const
  {
    UMeshField::ints<8> indices = hexIndices[hexID];
    return box4f()
      .including(make_vec4f(vertices[indices[0]]))
      .including(make_vec4f(vertices[indices[1]]))
      .including(make_vec4f(vertices[indices[2]]))
      .including(make_vec4f(vertices[indices[3]]))
      .including(make_vec4f(vertices[indices[4]]))
      .including(make_vec4f(vertices[indices[5]]))
      .including(make_vec4f(vertices[indices[6]]))
      .including(make_vec4f(vertices[indices[7]]));
  }

  inline __both__
  box4f UMeshField::DD::gridBounds(int gridID) const
  {
    return gridDomains[gridID];
  }
  
  inline __both__
  box4f UMeshField::DD::eltBounds(Element element) const
  {
    switch (element.type) {
    case Element::TET: 
      return tetBounds(element.ID);
    case Element::PYR:
      return pyrBounds(element.ID);
    case Element::WED:
      return wedBounds(element.ID);
    case Element::HEX: 
      return hexBounds(element.ID);
    case Element::GRID:
      return gridBounds(element.ID);
    default:
      return box4f(); 
    }
  }

  /*! evaluate (relative) distance of point P to the implicit plane
      defined by points A,B,C. distance is not normalized */
  inline __both__
  float evalToImplicitPlane(vec3f P, vec3f a, vec3f b, vec3f c)
  {
    vec3f N = cross(b-a,c-a);
    return dot(P-a,N);
  }

  /*! evaluate (relative) distance of point P to the implicit plane
      defined by points A,B,C. distance is not normalized */
  inline __both__
  float evalToImplicitPlane(vec3f P, vec4f a, vec4f b, vec4f c)
  { return evalToImplicitPlane(P,getPos(a),getPos(b),getPos(c)); }

  /*! evaluate (relative) distance of point P to the implicit plane
      defined by points A,B,C. distance is not normalized */
  inline __both__
  float evalToImplicitPlane(vec3f P, float4 a, float4 b, float4 c)
  { return evalToImplicitPlane(P,getPos(a),getPos(b),getPos(c)); }

  /* compute scalar of given umesh element at point P, and return that
     in 'retVal'. returns true if P is inside the elemnt, false if
     outside (in which case retVal is not defined) */
  inline __both__
  bool UMeshField::DD::eltScalar(float &retVal, Element elt, vec3f P) const
  {
    switch (elt.type) {
    case Element::TET: 
      return tetScalar(retVal,elt.ID,P);
    case Element::PYR:
      return pyrScalar(retVal,elt.ID,P);
    case Element::WED:
      return wedScalar(retVal,elt.ID,P);
    case Element::HEX:
      return hexScalar(retVal,elt.ID,P);
    case Element::GRID:
      return gridScalar(retVal,elt.ID,P);
    }
    return false;
  }
  
  inline __both__
  bool UMeshField::DD::tetScalar(float &retVal, int primID, vec3f P) const
  {
    int4 indices = tetIndices[primID];
    float4 v0 = vertices[indices.x];
    float4 v1 = vertices[indices.y];
    float4 v2 = vertices[indices.z];
    float4 v3 = vertices[indices.w];
    
    float t3 = evalToImplicitPlane(P,v0,v1,v2);
    if (t3 < 0.f) return false;
    float t2 = evalToImplicitPlane(P,v0,v3,v1);
    if (t2 < 0.f) return false;
    float t1 = evalToImplicitPlane(P,v0,v2,v3);
    if (t1 < 0.f) return false;
    float t0 = evalToImplicitPlane(P,v1,v3,v2);
    if (t0 < 0.f) return false;
    
    float scale = 1.f/(t0+t1+t2+t3);
    retVal = scale * (t0*v0.w + t1*v1.w + t2*v2.w + t3*v3.w);
    return true;
  }

  inline __both__
  bool UMeshField::DD::pyrScalar(float &retVal, int primID, vec3f P) const
  {
    const auto& indices = pyrIndices[primID];
    return intersectPyrEXT(retVal, P,
                           vertices[indices[0]],
                           vertices[indices[1]],
                           vertices[indices[2]],
                           vertices[indices[3]],
                           vertices[indices[4]]);
  }

  inline __both__
  bool UMeshField::DD::wedScalar(float &retVal, int primID, vec3f P) const
  {
    const auto& indices = wedIndices[primID];
    return intersectWedgeEXT(retVal, P,
                             vertices[indices[0]],
                             vertices[indices[1]],
                             vertices[indices[2]],
                             vertices[indices[3]],
                             vertices[indices[4]],
                             vertices[indices[5]]);
  }

  inline __both__
  bool UMeshField::DD::hexScalar(float &retVal, int primID, vec3f P) const
  {
    auto indices = hexIndices[primID];
    return intersectHexEXT(retVal, P,
                           vertices[indices[0]],
                           vertices[indices[1]],
                           vertices[indices[2]],
                           vertices[indices[3]],
                           vertices[indices[4]],
                           vertices[indices[5]],
                           vertices[indices[6]],
                           vertices[indices[7]]);
  }

  inline __both__
  bool UMeshField::DD::gridScalar(float &retVal, int primID, vec3f P) const
  {
    const box3f bounds = box3f((const vec3f &)gridDomains[primID].lower,
                               (const vec3f &)gridDomains[primID].upper);
    
    if (!bounds.contains(P))
      return false;

    vec3i numScalars = gridDims[primID]+1;
    vec3f cellSize = bounds.size()/vec3f(gridDims[primID]);
    vec3f objPos = (P-bounds.lower)/cellSize;
    vec3i imin(objPos);
    vec3i imax = min(imin+1,numScalars-1);

    auto linearIndex = [numScalars](const int x, const int y, const int z) {
                         return z*numScalars.y*numScalars.x + y*numScalars.x + x;
                       };

    const float *scalars = gridScalars + gridOffsets[primID];

    float f1 = scalars[linearIndex(imin.x,imin.y,imin.z)];
    float f2 = scalars[linearIndex(imax.x,imin.y,imin.z)];
    float f3 = scalars[linearIndex(imin.x,imax.y,imin.z)];
    float f4 = scalars[linearIndex(imax.x,imax.y,imin.z)];

    float f5 = scalars[linearIndex(imin.x,imin.y,imax.z)];
    float f6 = scalars[linearIndex(imax.x,imin.y,imax.z)];
    float f7 = scalars[linearIndex(imin.x,imax.y,imax.z)];
    float f8 = scalars[linearIndex(imax.x,imax.y,imax.z)];

#define EMPTY(x) isnan(x)
    if (EMPTY(f1) || EMPTY(f2) || EMPTY(f3) || EMPTY(f4) ||
        EMPTY(f5) || EMPTY(f6) || EMPTY(f7) || EMPTY(f8))
      return false;

    vec3f frac = objPos-vec3f(imin);

    float f12 = lerp(f1,f2,frac.x);
    float f56 = lerp(f5,f6,frac.x);
    float f34 = lerp(f3,f4,frac.x);
    float f78 = lerp(f7,f8,frac.x);

    float f1234 = lerp(f12,f34,frac.y);
    float f5678 = lerp(f56,f78,frac.y);

    retVal = lerp(f1234,f5678,frac.z);

    return true;
  }
  
}
