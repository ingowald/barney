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

  struct Element {
    typedef enum {
      TET=0, PYR, WED, HEX// , GRID
    } Type;
    
    inline __rtc_device Element() {}
    inline __rtc_device Element(int ofs0, Type type)
      : type(type), ofs0(ofs0)
    {}
    uint32_t ofs0:29;
    uint32_t type: 3;
  };

  struct UMeshField : public ScalarField
  {
    typedef std::shared_ptr<UMeshField> SP;

    virtual ~UMeshField()
    {
      PING;
      PING;
      PING;
      PING;
      PING;
      PING;
      exit(0);
    }
    
    /*! helper class for representing an N-long integer tuple, to
       represent wedge, pyramid, hex, etc elemnet indices */
    template<int N>
    struct ints { int v[N];
      inline __rtc_device int &operator[](int i)      { return v[i]; }
      inline __rtc_device int operator[](int i) const { return v[i]; }
    };
    /*! device-data for a unstructured-mesh scalar field, containing
        all device-side pointers and function to access this field and
        sample/evaluate its elemnets */
    struct DD : public ScalarField::DD {
      
      inline __rtc_device box4f eltBounds(Element element) const;
      inline __rtc_device box4f tetBounds(int ofs0) const;
      inline __rtc_device box4f pyrBounds(int ofs0) const;
      inline __rtc_device box4f wedBounds(int ofs0) const;
      inline __rtc_device box4f hexBounds(int ofs0) const;
      // inline __rtc_device box4f gridBounds(int ofs0) const;

      /* compute scalar of given umesh element at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __rtc_device bool eltScalar(float &retVal, Element elt, vec3f P, bool dbg = false) const;
      
      /* compute scalar of given tet in umesh, at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __rtc_device bool tetScalar(float &retVal, int ofs0, vec3f P, bool dbg = false) const;
      
      /* compute scalar of given pyramid in umesh, at point P, and
         return that in 'retVal'. returns true if P is inside the
         elemnt, false if outside (in which case retVal is not
         defined) */
      inline __rtc_device bool pyrScalar(float &retVal, int ofs0, vec3f P) const;
      
      /* compute scalar of given wedge in umesh, at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __rtc_device bool wedScalar(float &retVal, int ofs0, vec3f P) const;
      
      /* compute scalar of given hex in umesh, at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      inline __rtc_device
      bool hexScalar(float &retVal, int ofs0, vec3f P, bool dbg=false) const;
      
      /* compute scalar of given grid in umesh, at point P, and return
         that in 'retVal'. returns true if P is inside the elemnt,
         false if outside (in which case retVal is not defined) */
      // inline __rtc_device bool gridScalar(float &retVal, int ofs0, vec3f P) const;
      
      const rtc::float4 *vertices;
      const int         *indices;
      const Element     *elements;
      int                numElements;
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
    
    UMeshField(Context *context,
               const DevGroup::SP &devices);

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
    PODData::SP vertices;
    PODData::SP indices;
    PODData::SP elementOffsets;
    int numElements;
    /*! @} */
    /* internal-format 'elements' that encode both elemne type and
       offset in the indices array; each element is self-contained so
       can be re-ordered at will */
    // std::vector<Element>    elements;
    struct PLD {
      box3f   *pWorldBounds = 0;
      Element *elements     = 0;
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;
  };
  
  // ==================================================================
  // IMPLEMENTATION
  // ==================================================================
  
  inline __rtc_device
  box4f UMeshField::DD::tetBounds(int ofs0) const
  {
    const vec4i indices = *(const vec4i*)&this->indices[ofs0];
    return box4f()
      .including(rtc::load(vertices[indices.x]))
      .including(rtc::load(vertices[indices.y]))
      .including(rtc::load(vertices[indices.z]))
      .including(rtc::load(vertices[indices.w]));
  }

  inline __rtc_device
  box4f UMeshField::DD::pyrBounds(int ofs0) const
  {
    UMeshField::ints<5> indices = *(const UMeshField::ints<5> *)&this->indices[ofs0];
    return box4f()
      .including(rtc::load(vertices[indices[0]]))
      .including(rtc::load(vertices[indices[1]]))
      .including(rtc::load(vertices[indices[2]]))
      .including(rtc::load(vertices[indices[3]]))
      .including(rtc::load(vertices[indices[4]]));
  }

  inline __rtc_device
  box4f UMeshField::DD::wedBounds(int ofs0) const
  {
    UMeshField::ints<6> indices = *(const UMeshField::ints<6> *)&this->indices[ofs0];
    return box4f()
      .including(rtc::load(vertices[indices[0]]))
      .including(rtc::load(vertices[indices[1]]))
      .including(rtc::load(vertices[indices[2]]))
      .including(rtc::load(vertices[indices[3]]))
      .including(rtc::load(vertices[indices[4]]))
      .including(rtc::load(vertices[indices[5]]));
  }
  
  inline __rtc_device
  box4f UMeshField::DD::hexBounds(int ofs0) const
  {
    UMeshField::ints<8> indices = *(const UMeshField::ints<8> *)&this->indices[ofs0];
    return box4f()
      .including(rtc::load(vertices[indices[0]]))
      .including(rtc::load(vertices[indices[1]]))
      .including(rtc::load(vertices[indices[2]]))
      .including(rtc::load(vertices[indices[3]]))
      .including(rtc::load(vertices[indices[4]]))
      .including(rtc::load(vertices[indices[5]]))
      .including(rtc::load(vertices[indices[6]]))
      .including(rtc::load(vertices[indices[7]]));
  }
 
  inline __rtc_device
  box4f UMeshField::DD::eltBounds(Element element) const
  {
    switch (element.type) {
    case Element::TET: 
      return tetBounds(element.ofs0);
    case Element::PYR:
      return pyrBounds(element.ofs0);
    case Element::WED:
      return wedBounds(element.ofs0);
    case Element::HEX: 
      return hexBounds(element.ofs0);
    // case Element::GRID0:
    //   return gridBounds(element.ofs0);
    default:
      return box4f(); 
    }
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

  /*! evaluate (relative) distance of point P to the implicit plane
      defined by points A,B,C. distance is not normalized */
  // inline __rtc_device
  // float evalToImplicitPlane(vec3f P, float4 a, float4 b, float4 c)
  // { return evalToImplicitPlane(P,getPos(a),getPos(b),getPos(c)); }

  /* compute scalar of given umesh element at point P, and return that
     in 'retVal'. returns true if P is inside the elemnt, false if
     outside (in which case retVal is not defined) */
  inline __rtc_device
  bool UMeshField::DD::eltScalar(float &retVal,
                                 Element elt,
                                 vec3f P,
                                 bool dbg) const
  {
    switch (elt.type) {
    case Element::TET: 
      return tetScalar(retVal,elt.ofs0,P,dbg);
    case Element::PYR:
      return pyrScalar(retVal,elt.ofs0,P);
    case Element::WED:
      return wedScalar(retVal,elt.ofs0,P);
    case Element::HEX:
      return hexScalar(retVal,elt.ofs0,P,dbg);
    }
    return false;
  }
  
  inline __rtc_device
  bool UMeshField::DD::tetScalar(float &retVal, int ofs0, vec3f P, bool dbg) const
  {
    vec4i indices = *(const vec4i *)&this->indices[ofs0];
    
    vec4f v0 = rtc::load(vertices[indices.x]);
    vec4f v1 = rtc::load(vertices[indices.y]);
    vec4f v2 = rtc::load(vertices[indices.z]);
    vec4f v3 = rtc::load(vertices[indices.w]);

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

  inline __rtc_device
  bool UMeshField::DD::pyrScalar(float &retVal, int ofs0, vec3f P) const
  {
    UMeshField::ints<5> indices = *(const UMeshField::ints<5> *)&this->indices[ofs0];
    return intersectPyrEXT(retVal, P,
                           rtc::load(vertices[indices[0]]),
                           rtc::load(vertices[indices[1]]),
                           rtc::load(vertices[indices[2]]),
                           rtc::load(vertices[indices[3]]),
                           rtc::load(vertices[indices[4]]));
  }

  inline __rtc_device
  bool UMeshField::DD::wedScalar(float &retVal, int ofs0, vec3f P) const
  {
    UMeshField::ints<6> indices
      = *(const UMeshField::ints<6> *)&this->indices[ofs0];
    return intersectWedgeEXT(retVal, P,
                             rtc::load(vertices[indices[0]]),
                             rtc::load(vertices[indices[1]]),
                             rtc::load(vertices[indices[2]]),
                             rtc::load(vertices[indices[3]]),
                             rtc::load(vertices[indices[4]]),
                             rtc::load(vertices[indices[5]]));
  }

  inline __rtc_device
  bool UMeshField::DD::hexScalar(float &retVal,
                                 int ofs0,
                                 vec3f P,
                                 bool dbg) const
  {
    UMeshField::ints<8> indices
      = *(const UMeshField::ints<8> *)&this->indices[ofs0];
    return intersectHexEXT(retVal, P,
                           rtc::load(vertices[indices[0]]),
                           rtc::load(vertices[indices[1]]),
                           rtc::load(vertices[indices[2]]),
                           rtc::load(vertices[indices[3]]),
                           rtc::load(vertices[indices[4]]),
                           rtc::load(vertices[indices[5]]),
                           rtc::load(vertices[indices[6]]),
                           rtc::load(vertices[indices[7]]),
                           dbg);
  }

  
}
