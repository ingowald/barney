// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

#include "barney/ModelSlot.h"
/* all routines for point-element sampling/intersection - shold
   logically be part of this file, but kept in separate file because
   these were mostly imported from oepnvkl */
#include "barney/umesh/common/ElementIntersection.h"
#include "barney/volume/MCAccelerator.h"

namespace BARNEY_NS {

  enum {
    _ANARI_TET = 0,
    _ANARI_HEX = 1,
    _ANARI_PRISM = 2,
    _ANARI_PYR = 3
  };
  enum {
    _VTK_TET = 10,
    _VTK_HEX = 12,
    _VTK_PRISM = 13,
    _VTK_PYR = 14
  };
  
  struct UMeshField : public ScalarField
  {
    typedef std::shared_ptr<UMeshField> SP;

    UMeshField(Context *context,
               const DevGroup::SP &devices);

    virtual ~UMeshField()
    {}
    
    /*! helper class for representing an N-long integer tuple, to
       represent prism, pyramid, hex, etc elemnet indices */
    template<int N>
    struct ints { int v[N];
      inline __rtc_device int &operator[](int i)      { return v[i]; }
      inline __rtc_device int operator[](int i) const { return v[i]; }
    };
    /*! device-data for a unstructured-mesh scalar field, containing
        all device-side pointers and function to access this field and
        sample/evaluate its elemnets */
    struct DD : public ScalarField::DD {
      
      inline __rtc_device box4f cellBounds(uint32_t cellIdx) const;

      /* compute scalar of given umesh element at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __rtc_device bool eltScalar(float &retVal,
                                         uint32_t cellIdx,
                                         vec3f P,
                                         bool dbg = false) const;
      
      /* compute scalar of given tet in umesh, at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __rtc_device bool tetScalar(float &retVal,
                                         uint32_t cellIdx,
                                         vec3f P,
                                         bool dbg = false) const;
      
      /* compute scalar of given pyramid in umesh, at point P, and
         return that in 'retVal'. returns true if P is inside the
         elemnt, false if outside (in which case retVal is not
         defined) */
      inline __rtc_device bool pyrScalar(float &retVal,
                                         uint32_t cellIdx,
                                         vec3f P) const;
      
      /* compute scalar of given prism in umesh, at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __rtc_device bool prismScalar(float &retVal,
                                         uint32_t cellIdx,
                                         vec3f P) const;
      
      /* compute scalar of given hex in umesh, at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __rtc_device
      bool hexScalar(float &retVal,
                     uint32_t cellIdx,
                     vec3f P, bool
                     dbg=false) const;
      
      /* compute scalar of given grid in umesh, at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      // inline __rtc_device bool gridScalar(float &retVal, int ofs0, vec3f P) const;

      const vec3f       *vertices;
      const float       *scalars;
      const int         *indices;
      const int         *cellOffsets;
      const uint8_t     *cellTypes;
      int                numCells;
      bool               scalarsArePerVertex;
    };

    /*! build *initial* macro-cell grid (ie, the scalar field min/max
      ranges, but not yet the majorants) over a umesh */
    void buildInitialMacroCells(MCGrid &grid);

    /*! computes, on specified device, the bounding boxes and - if
      d_primRanges is non-null - the primitmives ranges. d_primBounds
      and d_primRanges (if non-null) must be pre-allocated and
      writeaable on specified device */
    void computeElementBBs(Device  *device,
                           box3f   *d_primBounds,
                           range1f *d_primRanges=0);
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool setData(const std::string &member,
                 const std::shared_ptr<Data> &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    
    DD getDD(Device *device);

    void buildMCs(MCGrid &macroCells) override;

    VolumeAccel::SP createAccel(Volume *volume) override;

    /*! returns part of the string used to find the optix device
        programs that operate on this type */
    static std::string typeName() { return "UMesh"; };

    /*! @{ set by the user, as paramters */
    PODData::SP scalars;
    PODData::SP indices;
    PODData::SP cellOffsets;
    PODData::SP/*uint8_t*/ cellTypes;
    PODData::SP vertices;
    int numCells;
    bool scalarsArePerVertex = false;
    /*! @} */
    struct PLD {
      box3f   *pWorldBounds = 0;
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;
  };
  
  // ==================================================================
  // IMPLEMENTATION
  // ==================================================================

  // inline __rtc_device void checkOrientation(int faceID,
  //                                      vec3f *hex,
  //                                      vec3f center,
  //                                      int ia, int ib, int ic, int id
  //                                      )
  // {
  //   vec3f a = hex[ia];
  //   vec3f b = hex[ib];
  //   vec3f c = hex[ic];
  //   vec3f d = hex[id];
  //   vec3f N0 = cross(b-a,c-a);
  //   vec3f N1 = cross(c-a,d-a);
  //   if (dot(N0,N0) == 0.f) printf("zero normal\n");
  //   if (dot(N1,N1) == 0.f) printf("zero normal\n");
  //   N0 = normalize(N0);
  //   N1 = normalize(N1);
  //   if (dot(N0,N1) < .98f) printf("broken face\n");
  //   if (dot(center-a,N0) < 0.f) printf("correct face orientation\n");
  //   // if (dot(center-a,N0) > 0.f) printf("broken face orientation\n");
  // }
  
  
  // inline __rtc_device void checkOrientation(vec3f *hex)
  // {
  //   vec3f center = 1.f/8.f*(hex[0]+hex[1]+
  //                           hex[2]+hex[3]+
  //                           hex[4]+hex[5]+
  //                           hex[6]+hex[7]);
  //   checkOrientation(0,hex,center,0,1,5,4);
  //   checkOrientation(1,hex,center,1,2,6,5);
  //   checkOrientation(2,hex,center,2,3,7,6);
  //   checkOrientation(3,hex,center,3,0,4,7);
  //   checkOrientation(4,hex,center,4,5,6,7);
  //   checkOrientation(5,hex,center,0,3,2,1);
  // }
  
  
  inline __rtc_device
  box4f UMeshField::DD::cellBounds(uint32_t cellIdx) const
  {
    uint32_t cellType = cellTypes[cellIdx];
    uint32_t numVertices = 0;
    uint32_t offset = cellOffsets[cellIdx];
    switch (cellType) {
    case _VTK_TET:
    case _ANARI_TET:
      numVertices = 4;
      break;
    case _VTK_PYR:
    case _ANARI_PYR:
      numVertices = 5;
      break;
    case _VTK_PRISM:
    case _ANARI_PRISM:
      numVertices = 6;
      break;
    case _VTK_HEX: 
    case _ANARI_HEX: 
      numVertices = 8;
      break;
    default:
      ;
    }
// #if 1
//     if (numVertices == 8) {
//       vec3f vv[8];
//       for (uint32_t i=0;i<numVertices;i++) {
//         int vtxIdx = indices[offset++];
//         // vec4f v(vertices[vtxIdx],scalars[scalarsArePerVertex?vtxIdx:cellIdx]);
//         vv[i] = vertices[vtxIdx];//v;
//       }
//       checkOrientation(vv);
//     }
// #endif
    
    box4f bb;
    for (uint32_t i=0;i<numVertices;i++) {
      int vtxIdx = indices[offset++];
      vec4f v(vertices[vtxIdx],scalars[scalarsArePerVertex?vtxIdx:cellIdx]);
      bb.extend(v);
    }
    return bb;
  }

  /*! evaluate (relative) distance of point P to the implicit plane
      defined by points A,B,C. distance is not normalized */
  inline __rtc_device
  float evalToImplicitPlane(vec3f P, vec3f a, vec3f b, vec3f c)
  {
    vec3f N = cross(b-a,c-a);
    return dot(P-a,N);
  }

  /*! evaluate (relative) distance of point P to the implicit plane
      defined by points A,B,C. distance is not normalized */
  inline __rtc_device
  float evalToImplicitPlane(vec3f P, vec4f a, vec4f b, vec4f c)
  { return evalToImplicitPlane(P,getPos(a),getPos(b),getPos(c)); }

  /* compute scalar of given umesh element at point P, and return that
     in 'retVal'. returns true if P is inside the elemnt, false if
     outside (in which case retVal is not defined) */
  inline __rtc_device
  bool UMeshField::DD::eltScalar(float &retVal,
                                 uint32_t cellIdx,
                                 vec3f P,
                                 bool dbg) const
  {
    uint8_t cellType = cellTypes[cellIdx];
    switch (cellType) {
    case _ANARI_TET: 
    case _VTK_TET: 
      return tetScalar(retVal,cellIdx,P,dbg);
    case _ANARI_PYR:
    case _VTK_PYR:
      return pyrScalar(retVal,cellIdx,P);
    case _ANARI_PRISM:
    case _VTK_PRISM:
      return prismScalar(retVal,cellIdx,P);
    case _ANARI_HEX:
    case _VTK_HEX:
      return hexScalar(retVal,cellIdx,P,dbg);
    }
    return false;
  }
  
  inline __rtc_device
  bool UMeshField::DD::tetScalar(float &retVal,
                                 uint32_t cellIdx,
                                 vec3f P,
                                 bool dbg) const
  {
    int ofs0 = cellOffsets[cellIdx];
    vec4i indices = *(const vec4i *)&this->indices[ofs0];
    
    vec4f v0(vertices[indices.x],scalars[scalarsArePerVertex?indices.x:cellIdx]);
    vec4f v1(vertices[indices.y],scalars[scalarsArePerVertex?indices.y:cellIdx]);
    vec4f v2(vertices[indices.z],scalars[scalarsArePerVertex?indices.z:cellIdx]);
    vec4f v3(vertices[indices.w],scalars[scalarsArePerVertex?indices.w:cellIdx]);

    float t3 = evalToImplicitPlane(P,v0,v1,v2);
    if (t3 < 0.f) return false;
    float t2 = evalToImplicitPlane(P,v0,v3,v1);
    if (t2 < 0.f) return false;
    float t1 = evalToImplicitPlane(P,v0,v2,v3);
    if (t1 < 0.f) return false;
    float t0 = evalToImplicitPlane(P,v1,v3,v2);
    if (t0 < 0.f) return false;

    if (scalarsArePerVertex) {
      float scale = 1.f/(t0+t1+t2+t3);
      retVal = scale * (t0*v0.w + t1*v1.w + t2*v2.w + t3*v3.w);
    } else {
      retVal = scalars[cellIdx];
    }
    return true;
  }
  
  inline __rtc_device
  bool UMeshField::DD::pyrScalar(float &retVal,
                                 uint32_t cellIdx,
                                 vec3f P) const
  {
    int ofs0 = cellOffsets[cellIdx];
    UMeshField::ints<5> indices = *(const UMeshField::ints<5> *)&this->indices[ofs0];
    vec4f v0(vertices[indices[0]],scalars[scalarsArePerVertex?indices[0]:cellIdx]);
    vec4f v1(vertices[indices[1]],scalars[scalarsArePerVertex?indices[1]:cellIdx]);
    vec4f v2(vertices[indices[2]],scalars[scalarsArePerVertex?indices[2]:cellIdx]);
    vec4f v3(vertices[indices[3]],scalars[scalarsArePerVertex?indices[3]:cellIdx]);
    vec4f v4(vertices[indices[4]],scalars[scalarsArePerVertex?indices[4]:cellIdx]);
    return intersectPyrEXT(retVal, P, v0,v1,v2,v3,v4);
  }

  inline __rtc_device
  bool UMeshField::DD::prismScalar(float &retVal,
                                 uint32_t cellIdx,
                                 vec3f P) const
  {
    int ofs0 = cellOffsets[cellIdx];
    UMeshField::ints<6> indices
      = *(const UMeshField::ints<6> *)&this->indices[ofs0];
    vec4f v0(vertices[indices[0]],scalars[scalarsArePerVertex?indices[0]:cellIdx]);
    vec4f v1(vertices[indices[1]],scalars[scalarsArePerVertex?indices[1]:cellIdx]);
    vec4f v2(vertices[indices[2]],scalars[scalarsArePerVertex?indices[2]:cellIdx]);
    vec4f v3(vertices[indices[3]],scalars[scalarsArePerVertex?indices[3]:cellIdx]);
    vec4f v4(vertices[indices[4]],scalars[scalarsArePerVertex?indices[4]:cellIdx]);
    vec4f v5(vertices[indices[5]],scalars[scalarsArePerVertex?indices[5]:cellIdx]);
    return intersectPrismEXT(retVal, P, v0,v1,v2,v3,v4,v5);
  }

  inline __rtc_device
  bool UMeshField::DD::hexScalar(float &retVal,
                                 uint32_t cellIdx,
                                 vec3f P,
                                 bool dbg) const
  {
    int ofs0 = cellOffsets[cellIdx];
    UMeshField::ints<8> indices
      = *(const UMeshField::ints<8> *)&this->indices[ofs0];
    vec4f v0(vertices[indices[0]],scalars[scalarsArePerVertex?indices[0]:cellIdx]);
    vec4f v1(vertices[indices[1]],scalars[scalarsArePerVertex?indices[1]:cellIdx]);
    vec4f v2(vertices[indices[2]],scalars[scalarsArePerVertex?indices[2]:cellIdx]);
    vec4f v3(vertices[indices[3]],scalars[scalarsArePerVertex?indices[3]:cellIdx]);
    vec4f v4(vertices[indices[4]],scalars[scalarsArePerVertex?indices[4]:cellIdx]);
    vec4f v5(vertices[indices[5]],scalars[scalarsArePerVertex?indices[5]:cellIdx]);
    vec4f v6(vertices[indices[6]],scalars[scalarsArePerVertex?indices[6]:cellIdx]);
    vec4f v7(vertices[indices[7]],scalars[scalarsArePerVertex?indices[7]:cellIdx]);
    return intersectHexEXT(retVal, P, v0,v1,v2,v3,v4,v5,v6,v7, dbg);
  }
  
}
