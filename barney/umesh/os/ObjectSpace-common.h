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

#include "barney/umesh/common/UMeshField.h"
#include "barney/common/CUBQL.h"
#include "barney/render/Ray.h"

namespace barney {
  using render::Ray;
  using render::boxTest;
  using ints5 = UMeshField::ints<5>;
  using ints6 = UMeshField::ints<6>;
  using ints8 = UMeshField::ints<8>;
  
  // ------------------------------------------------------------------
  /*! base class for object-space volume accelerator for an
    unstructured mesh field */
  struct UMeshObjectSpace {
    struct DD : public UMeshField::DD {
      using Inherited = UMeshField::DD;
      
      // static void addVars(std::vector<OWLVarDecl> &vars, int base)
      // {
      //   Inherited::addVars(vars,base);
      //   TransferFunction::DD::addVars(vars,base+OWL_OFFSETOF(DD,xf));
      // }

      TransferFunction::DD xf;
    };

    struct Host : public VolumeAccel {
      Host(UMeshField *mesh, Volume *volume)
        : VolumeAccel(mesh,volume),
          mesh(mesh)
      {}

      UpdateMode updateMode() override
      { return HAS_ITS_OWN_GROUP; }
      
      /*! set owl variables for this accelerator - this is virutal so
        derived classes can add their own */
      // virtual void setVariables(OWLGeom geom)
      // {
      //   mesh->setVariables(geom);
      //   getXF()->setVariables(geom);
      // }
      void writeDD(UMeshObjectSpace::DD &dd, Device *device)
      {
        mesh->writeDD(dd,device);
      }
      
      UMeshField *const mesh;
      OWLGeom   geom  = 0;
      OWLGroup  group = 0;
    };
  };
  
  /*! the main function provided by this header file - intersecting a
    ray against a given leaf node in a object-space umesh bvh */
  inline __device__
  float intersectLeaf(Ray &ray,
                      range1f &inputLeafRange,
                      const UMeshObjectSpace::DD &self,
                      int begin,
                      int end, bool dbg = false);

  // ------------------------------------------------------------------
  /*! helper class that represents a ray segment (from begin.t to
    end.t) whose scalar values is linearly varying from begin.scalar
    (at begin.t) to end.scalar (at end.t) */
  struct LinearSegment {
    struct EndPoint { float t, scalar; };

    /*! compute scalar range covered by this segment */
    __device__ range1f scalarRange() const;

    /*! returns the interpolated scalar value along this segment; this
      code assumes that t is INSIDE that segment! */
    __device__ float lerp(float t) const;
    
    EndPoint begin, end;
  };

  // ------------------------------------------------------------------
  /*! helper class that represents a plane equation defiend through
    three points; can be used to clip a ray segment or evaluate
    distance to a point */
  struct Plane {
    /*! 'constructor' the sets a new plane equation going through
      three points */
    __device__ void set(vec3f a, vec3f b, vec3f c);

    /*! clip given linear segment to the POSITIVE half-space defined
      by this plane (ie, for a tet this is assuming INSIDE-facing
      planes like VTK) */
    __device__ bool clip(LinearSegment &segment,
                         const Ray &ray) const;

    /*! compute (scaled) distance of point v to this plane */
    __device__ float eval(vec3f v) const;
    
    vec3f N; //!< normal of plane
    vec3f P; //!< anchor point of plane
  };

  // ------------------------------------------------------------------
  /*! does first step of object-space intersection between a tet and a
    ray; generating a LinearSegment that describes the scalar
    function along the [t0,t1] interavl that the ray overlaps this
    tet (which for a tet must be a linear function) */
  struct TetIntersector {
    __device__ bool set(const UMeshObjectSpace::DD &dd,
                        Element elt);
    __device__ void set(float4 v0, float4 v1, float4 v2, float4 v3);
    
    __device__ bool computeSegment(LinearSegment &segment,
                                   const Ray &ray,
                                   const range1f &inputRange) const;
    
    /*! compute interpolated scalar value at position P, assuming that
      P is inside the tet */
    __device__ float lerp_inside(const vec3f P) const;

    /*! the four positions of that tet's vertices */
    vec3f v0, v1, v2, v3;
    
    /*! the scalar values of that tet's four vertices */
    float w0, w1, w2, w3;
    
    /*! the four planes defined by the tet's four faces - p0 being the
      pane opposite v0, etc */
    Plane p0, p1, p2, p3;
  };
  

  /*! helper class that performs object--space ray-elemnt intersection
    for _general_ elements (pyrs, weds, and hexes); grid and tets are
    handled in separate cases */
  struct ElementIntersector
  {
    inline __device__
    ElementIntersector(const UMeshObjectSpace::DD &dd,
                       Ray &ray,
                       range1f leafRange);

    /*! compute curently set element's scalar value at P, map it
      through the trnasfer function, and store result in
      this->mapped; */
    inline __device__
    void sampleAndMap(vec3f P, bool dbg);

    /*! assume a enw element, and do whatever precomputations are
      required for that, deending on its type */
    inline __device__
    bool setElement(Element elt);
    
    /*! compute (scaled) distance of P relative to plane abc */
    inline __device__
    float evalToPlane(vec3f P, vec4f a, vec4f b, vec4f c);

    /*! clip currently active t-range (whch eventually describes
      interval of t values for which element overlaps the ray) to
      given plane, assuming inside facing planes */
    inline __device__
    void clipRangeToPlane(vec4f a, vec4f b, vec4f c);
    
    /*! clip currently active t-range (whch eventually describes
      interval of t values for which element overlaps the ray) to
      given bilinear path, assuming inside facing planes */
    inline __device__
    void clipRangeToPatch(vec4f a, vec4f b, vec4f c, vec4f d);

    /*! evaluate the current element at position P, storing the scalar
      value in sample, and returnign whether the point was actually
      inside (true) or outside (false) the volume; if outside the
      retured sample may be undefined */
    inline __device__ bool evalElement(vec3f P, float &sample);

    /*! 'evalElement' for a specific tet */
    inline __device__
    bool evalTet(vec3f P, float &sample);

    /*! 'evalElement' for a specific pyr elemnt */
    inline __device__
    bool evalPyr(vec3f P, float& sample);

    /*! 'evalElement' for a specific wedge element */
    inline __device__
    bool evalWed(vec3f P, float& sample);

    /*! 'evalElement' for a specific hex element */
    inline __device__
    bool evalHex(vec3f P, float& sample);

    /*! comute (conservative) *scalar* range of current ray-elemtn
      overlap segmehnt; this can be conservative, so may actually
      compute the entire element's scalr range */
    inline __device__
    range1f computeElementScalarRange();

    /*! computes and sets the current element's t range (ie,
      this->elemnetTRange) of where it overlaps the ray. note this
      would be allowed to be conservative as long as it doesn't
      extend beyond the current ray.tMax/t_hit value - this elemnt
      is suppsoed to dfine the range of where we need to do woodcock
      stepping to sample that element at; if it is conservative that
      is OK as long as the evaluation code for such
      outside-the-elment positions would properly reutrn 'invalid'
      samples. (though obviously, the tighter the range the
      better) */
    inline __device__
    bool computeElementRange();

    /*! compute majorant of current t range */
    inline __device__
    float computeRangeMajorant();
    
    Ray &ray;

    // current t positoin's scalar and mapped avlues
    float scalar;
    vec4f mapped;
    
    // parameter interval of where ray has valid overlap with the
    // leaf's bbox (ie, where ray overlaps the box, up to tMax/tHit
    range1f leafRange;
    
    // paramter interval where ray's valid range overlaps the current *element*
    range1f elementTRange;
    
    /*! current unstructured element: */
    Element element;

    /*! the (up to) 8 vertices of the current element */
    vec4f v0, v1, v2, v3, v4, v5, v6, v7;

    const UMeshObjectSpace::DD &dd;
  };
  
  /*! helper class for object-space ray/leaf intersection - this
    computes routines for intersecting the ray with all tets, with
    all grids, and with all other general unstructured elemnets;
    internally using the TetIntesector (for tets) and the
    ElementIntersector (for all non-tet and non-grid elements. For
    tets this uses an otpimiation where it first "gathers" multiple
    different tets' LinearSegments, before doing the actual
    per-segment woodcock stepping; this aims at allowing differen
    threads to gather segments from potentially different leaf
    positions before embarking ont he sampling */
  struct NewIntersector {
    enum { maxSegments = 4 };

    const bool dbg;
    inline __device__
    NewIntersector(Ray &ray,
                   range1f &inputLeafRange,
                   const UMeshObjectSpace::DD &self,
                   int begin,
                   int end,
                   bool dbg=false);

    /*! intesect all test in currelt leaf, by gathering segments and
      intersecting those when required */
    inline __device__ void doTets();
    /*! do all non-tet-non-grid elemnets in current leaf, using
      specialized ElementIntersector helper class */
    inline __device__ void doOthers();
    /*! intesect all gridlets in currelt leaf */
    // inline __device__ void doGrids();
    
    /*! intersect all currently gathered segments - this is a helper
      function for doTets() */
    inline __device__ void doSegments();


    /* do ray-gridlet element intersection; currently doing only
       woodcock marching directly on that gridlets, using the entire
       gridlet's majorant. TODO: at some point compute how many
       samples to expect for full gridlet, and if too high, swtich to
       DDA and per-cell intersection */
    // inline __device__
    // void doGrid(int gridID);

    /*! sample a gridlet on given position */
    inline __device__
    float sampleGrid(const box4f &domain, vec3i dims, const float *scalars,
                     vec3f P);

    bool hadAnyGrids = false;
    bool hadAnyOthers = false;
    int numSegments = 0;
    LinearSegment segments[maxSegments];
    const UMeshObjectSpace::DD self;
    Ray &ray;
    range1f &inputLeafRange;
    const int begin;
    const int end;
    float hit_t = BARNEY_INF;
    LCG<4> &rand;
  };
  




  // ==================================================================
  // INLINE IMPLEMENTATION SECTION
  // ==================================================================
  

  // ------------------------------------------------------------------
  // any unsorted/general stuff
  // ------------------------------------------------------------------

  inline box3f make_box(cuBQL::box3f b)
  {
    return (const box3f&)b;
  }
  
  inline box4f make_box4f(box3f b, range1f r=range1f())
  {
    box4f b4;
    (vec3f&)b4.lower = (const vec3f&)b.lower;
    (vec3f&)b4.upper = (const vec3f&)b.upper;
    b4.lower.w = r.lower;
    b4.upper.w = r.upper;
    return b4;
  }
  
  
  // ------------------------------------------------------------------
  // "Plane::" stuff
  // ------------------------------------------------------------------
  
  inline __device__ void Plane::set(vec3f a, vec3f b, vec3f c)
  { N = cross(b-a,c-a); P = a; }
  
  inline __device__ float Plane::eval(vec3f v) const
  { return dot(v-P,N); }
  
  inline __device__
  bool Plane::clip(LinearSegment &segment,
                   const Ray &ray)
    const
  {
    float NdotD = dot(N,ray.dir);
                    
    if (NdotD == 0.f) { 
      if (eval(ray.org) <= 0.f) {
        segment.begin.t = BARNEY_INF;
        return false;
      }
      return true;
    }

    float plane_t = dot(P - ray.org, N) / NdotD;
    if (NdotD < 0.f)
      segment.end.t = min(segment.end.t,plane_t);
    else
      segment.begin.t = max(segment.begin.t,plane_t);

    return segment.begin.t < segment.end.t;
  }


  // ------------------------------------------------------------------
  // "LinearSegment::" stuff
  // ------------------------------------------------------------------
  inline __device__
  bool clipSegment(LinearSegment &clipped,
                   const LinearSegment &original,
                   float t,
                   bool dbg=false)
  {
    // if (dbg) printf("clipping segment %f %f to %f\n",original.begin.t,original.end.t,t);
    if (t < original.begin.t) return false;
    
    clipped = original;
    if (t >= clipped.end.t) {
      // nothing to do
    } else {
      clipped.end.scalar = original.lerp(t);
      clipped.end.t = t;
      // if (dbg)
      //   printf("clipped new end %f scalar %f\n",
      //          clipped.end.t,clipped.end.scalar);
    }
    return clipped.end.t >= clipped.begin.t;
  }

  /*! returns the interpolated scalar value along this segment; this
    code assumes that t is INSIDE that segment! */
  inline __device__
  float LinearSegment::lerp(float t) const
  {
    const float len = this->end.t-this->begin.t;
    if (len == 0.f) return this->begin.scalar;
    
    const float f = (t-this->begin.t)/len;
    return 
      (1.f-f)*this->begin.scalar
      +    f *this->end.scalar;
  }
    
  inline __device__
  range1f LinearSegment::scalarRange() const
  { return { min(begin.scalar,end.scalar),max(begin.scalar,end.scalar) }; }

  // ------------------------------------------------------------------
  // "TetIntersector::" stuff
  // ------------------------------------------------------------------

  inline __device__
  float TetIntersector::lerp_inside(const vec3f P) const
  {
    float t0 = p0.eval(P);
    float t1 = p1.eval(P);
    float t2 = p2.eval(P);
    float t3 = p3.eval(P);

    t0 = max(t0,0.f);
    t1 = max(t1,0.f);
    t2 = max(t2,0.f);
    t3 = max(t3,0.f);
    
    return
      (t0*w0 + t1*w1 + t2*w2 + t3*w3)
      /
      (t0+t1+t2+t3);
  }
  
  inline __device__
  void TetIntersector::set(float4 __v0,
                           float4 __v1,
                           float4 __v2,
                           float4 __v3)
  {
    w0 = __v0.w;
    w1 = __v1.w;
    w2 = __v2.w;
    w3 = __v3.w;
    v0 = getPos(__v0);
    v1 = getPos(__v1);
    v2 = getPos(__v2);
    v3 = getPos(__v3);
    
    p3.set(v0,v1,v2);
    p2.set(v0,v3,v1);
    p1.set(v0,v2,v3);
    p0.set(v1,v3,v2);
  }
  
  inline __device__
  bool TetIntersector::set(const UMeshObjectSpace::DD &dd,
                           Element elt)
  {
    if (elt.type != Element::TET) return false;

    int4 indices = (const vec4i &)dd.indices[elt.ofs0];
    set(dd.vertices[indices.x],
        dd.vertices[indices.y],
        dd.vertices[indices.z],
        dd.vertices[indices.w]);

    return true;
  }

  inline __device__
  bool TetIntersector::computeSegment(LinearSegment &segment,
                                      const Ray &ray,
                                      const range1f &inputRange)
    const
  {
    segment.begin.t = inputRange.lower;
    segment.end.t   = inputRange.upper;
    
    if (!p0.clip(segment,ray)) return false;
    if (!p1.clip(segment,ray)) return false;
    if (!p2.clip(segment,ray)) return false;
    if (!p3.clip(segment,ray)) return false;

    vec3f P_begin = ray.org + segment.begin.t * ray.dir;
    vec3f P_end   = ray.org + segment.end.t   * ray.dir;

    segment.begin.scalar = lerp_inside(P_begin);
    segment.end.scalar   = lerp_inside(P_end);
    return true;
  }

  // ------------------------------------------------------------------
  // "ElementIntersector::" stuff
  // ------------------------------------------------------------------
  inline __device__
  void ElementIntersector::sampleAndMap(vec3f P, bool dbg)
  {
    if (!evalElement(P, scalar))
      mapped = 0.f;
    else
      mapped = dd.xf.map(scalar,dbg);
  }


  inline __device__
  ElementIntersector::ElementIntersector(const UMeshObjectSpace::DD &dd,
                                         Ray &ray,
                                         range1f leafRange)
    : dd(dd), ray(ray), leafRange(leafRange)
  {}


  inline __device__
  bool ElementIntersector::setElement(Element elt)
  {
    element = elt;
    switch (elt.type)
      {
      case Element::TET: {
        vec4i indices = (const vec4i&)dd.indices[elt.ofs0];
        v0 = dd.vertices[indices.x];
        v1 = dd.vertices[indices.y];
        v2 = dd.vertices[indices.z];
        v3 = dd.vertices[indices.w];
      }
        break;
      case Element::PYR: {
        ints5 indices = (const ints5&)dd.indices[elt.ofs0];
        v0 = dd.vertices[indices[0]];
        v1 = dd.vertices[indices[1]];
        v2 = dd.vertices[indices[2]];
        v3 = dd.vertices[indices[3]];
        v4 = dd.vertices[indices[4]];
      }
        break;
      case Element::WED: {
        ints6 indices = (const ints6&)dd.indices[elt.ofs0];
        v0 = dd.vertices[indices[0]];
        v1 = dd.vertices[indices[1]];
        v2 = dd.vertices[indices[2]];
        v3 = dd.vertices[indices[3]];
        v4 = dd.vertices[indices[4]];
        v5 = dd.vertices[indices[5]];
      }
        break;
      case Element::HEX: {
        ints8 indices = (const ints8&)dd.indices[elt.ofs0];
        v0 = dd.vertices[indices[0]];
        v1 = dd.vertices[indices[1]];
        v2 = dd.vertices[indices[2]];
        v3 = dd.vertices[indices[3]];
        v4 = dd.vertices[indices[4]];
        v5 = dd.vertices[indices[5]];
        v6 = dd.vertices[indices[6]];
        v7 = dd.vertices[indices[7]];
      }
        break;
      default:
        return false;
      }
    return true;
  }

  /*! compute (scaled) distance of P relative to plane abc */
  inline __device__
  float ElementIntersector::evalToPlane(vec3f P, vec4f a, vec4f b, vec4f c)
  {
    vec3f N = (cross(getPos(b)-getPos(a),getPos(c)-getPos(a)));
    return dot(P-getPos(a),N);
  }

  /*! clip currently active t-range (whch eventually describes
    interval of t values for which element overlaps the ray) to
    given plane, assuming inside facing planes */
  inline __device__
  void ElementIntersector::clipRangeToPlane(vec4f a, vec4f b, vec4f c)
  {
    vec3f N = (cross(getPos(b)-getPos(a),getPos(c)-getPos(a)));
    float NdotD = dot((vec3f)ray.dir, N);
    if (NdotD == 0.f)
      return;
    float plane_t = dot(getPos(a) - ray.org, N) / NdotD;
    if (NdotD < 0.f)
      elementTRange.upper = min(elementTRange.upper,plane_t);
    else
      elementTRange.lower = max(elementTRange.lower,plane_t);
  }

  /*! clip currently active t-range (whch eventually describes
    interval of t values for which element overlaps the ray) to
    given bilinear path, assuming inside facing planes */
  inline __device__
  void ElementIntersector::clipRangeToPatch(vec4f a, vec4f b, vec4f c, vec4f d)
  {
    vec3f NRef = cross(getPos(b)-getPos(a),getPos(c)-getPos(a));
    vec3f ad = getPos(d) - getPos(a);

    if (dot(NRef, ad) >= 0){
      // abc - acd
      clipRangeToPlane(a, b, c);
      clipRangeToPlane(a, c, d);
    } else {
      // abd - bcd
      clipRangeToPlane(a, b, d);
      clipRangeToPlane(b, c, d);
    }
  }

  /*! evaluate the current element at position P, storing the scalar
    value in sample, and returnign whether the point was actually
    inside (true) or outside (false) the volume; if outside the
    retured sample may be undefined */
  inline __device__ bool ElementIntersector::evalElement(vec3f P, float &sample)
  {
    switch(element.type){
    case Element::Type::TET:
      return evalTet(P, sample);
    case Element::Type::PYR:
      return evalPyr(P, sample);
    case Element::Type::WED:
      return evalWed(P, sample);
    case Element::Type::HEX:
      return evalHex(P, sample);
    default:
      return false;
    }
  }

  /*! 'evalElement' for a specific tet */
  inline __device__
  bool ElementIntersector::evalTet(vec3f P, float &sample)
  {
    float t3 = evalToImplicitPlane(P,v0,v1,v2);
    if (t3 < 0.f) return false;
    float t2 = evalToImplicitPlane(P,v0,v3,v1);
    if (t2 < 0.f) return false;
    float t1 = evalToImplicitPlane(P,v0,v2,v3);
    if (t1 < 0.f) return false;
    float t0 = evalToImplicitPlane(P,v1,v3,v2);
    if (t0 < 0.f) return false;
    float scale = 1.f/(t0+t1+t2+t3);
    sample = scale * (t0*v0.w + t1*v1.w + t2*v2.w + t3*v3.w);
    return true;
  }

  /*! 'evalElement' for a specific pyr elemnt */
  inline __device__
  bool ElementIntersector::evalPyr(vec3f P, float& sample)
  {
    return intersectPyrEXT(sample, P, v0, v1, v2, v3, v4);
  }

  /*! 'evalElement' for a specific wedge element */
  inline __device__
  bool ElementIntersector::evalWed(vec3f P, float& sample)
  {
    return intersectWedgeEXT(sample, P, v0, v1, v2, v3, v4, v5);
  }

  /*! 'evalElement' for a specific hex element */
  inline __device__
  bool ElementIntersector::evalHex(vec3f P, float& sample)
  {
    return intersectHexEXT(sample, P, v0, v1, v2, v3, v4, v5, v6, v7);
  }

  /*! comute (conservative) *scalar* range of current ray-elemtn
    overlap segmehnt; this can be conservative, so may actually
    compute the entire element's scalr range */
  inline __device__
  range1f ElementIntersector::computeElementScalarRange() 
  {
    float scalar_t0 = 0.f;
    float scalar_t1 = 0.f;
    switch (element.type)
      {
        // tets currently already handled in separate Tet intersector
        // case Element::TET:
        //   evalTet(ray.org+elementTRange.lower*ray.dir, scalar_t0);
        //   evalTet(ray.org+elementTRange.upper*ray.dir, scalar_t1);
        //   return { min(scalar_t0,scalar_t1),max(scalar_t0,scalar_t1) };
      case Element::PYR:
        scalar_t0 = min(v0.w, min(v1.w, min(v2.w, min(v3.w, v4.w))));
        scalar_t1 = max(v0.w, max(v1.w, max(v2.w, max(v3.w, v4.w))));
        return {scalar_t0, scalar_t1};
      case Element::WED:
        scalar_t0 = min(v0.w, min(v1.w, min(v2.w, min(v3.w, min(v4.w, v5.w)))));
        scalar_t1 = max(v0.w, max(v1.w, max(v2.w, max(v3.w, max(v4.w, v5.w)))));
        return {scalar_t0, scalar_t1};
      case Element::HEX:
        scalar_t0 = min(min(v0.w, min(v1.w, min(v2.w, v3.w))),
                        min(v4.w, min(v5.w, min(v6.w, v7.w))));
          
        scalar_t1 = max(max(v0.w, max(v1.w, max(v2.w, v3.w))),
                        max(v4.w, max(v5.w, max(v6.w, v7.w))));
        return {scalar_t0, scalar_t1};
      }
    return range1f();
  }

  /*! computes and sets the current element's t range (ie,
    this->elemnetTRange) of where it overlaps the ray. note this
    would be allowed to be conservative as long as it doesn't
    extend beyond the current ray.tMax/t_hit value - this elemnt
    is suppsoed to dfine the range of where we need to do woodcock
    stepping to sample that element at; if it is conservative that
    is OK as long as the evaluation code for such
    outside-the-elment positions would properly reutrn 'invalid'
    samples. (though obviously, the tighter the range the
    better) */
  inline __device__
  bool ElementIntersector::computeElementRange()
  {
    elementTRange = leafRange;
    switch (element.type)
      {
        // tets currently already handled in separate Tet intersector
        // case Element::TET:
        //   clipRangeToPlane(v0,v1,v2);
        //   clipRangeToPlane(v0,v3,v1);
        //   clipRangeToPlane(v0,v2,v3);
        //   clipRangeToPlane(v1,v3,v2);
        //   break;
      case Element::PYR:
        clipRangeToPlane(v0, v4, v1);
        clipRangeToPlane(v0, v3, v4);
        clipRangeToPlane(v1, v4, v2);
        clipRangeToPlane(v2, v4, v3);
        clipRangeToPatch(v0, v1, v2, v3);
        break;
      case Element::WED:
        clipRangeToPlane(v0,v2,v1);
        clipRangeToPlane(v3,v4,v5);
        clipRangeToPatch(v0, v3, v5, v2);
        clipRangeToPatch(v0, v1, v4, v3);
        clipRangeToPatch(v1, v2, v5, v4);
        break;
      case Element::HEX:
        clipRangeToPatch(v0, v1, v2, v3);
        clipRangeToPatch(v0, v3, v7, v4);
        clipRangeToPatch(v0, v4, v5, v1);
        clipRangeToPatch(v6, v2, v1, v5);
        clipRangeToPatch(v6, v5, v4, v7);
        clipRangeToPatch(v6, v7, v3, v2);
        break;
      }
    return !elementTRange.empty();
  }
  
  /*! compute majorant of current t range */
  inline __device__
  float ElementIntersector::computeRangeMajorant()
  {
    return dd.xf.majorant(computeElementScalarRange());
  }


  // ------------------------------------------------------------------
  // "NewIntersector::" stuff
  // ------------------------------------------------------------------
  inline __device__
  NewIntersector::NewIntersector(Ray &ray,
                                 range1f &inputLeafRange,
                                 const UMeshObjectSpace::DD &self,
                                 int begin,
                                 int end,
                                 bool dbg)
    : ray(ray), self(self), inputLeafRange(inputLeafRange),
      begin(begin), end(end), rand((LCG<4> &)ray.rngSeed),
      dbg(dbg)

  {
    hit_t = ray.tMax;
    doTets();
    // if (hadAnyGrids)
    //   doGrids();
    if (hadAnyOthers)
      doOthers();
  }

  inline __device__
  void NewIntersector::doSegments()
  {
    // if (dbg)
    //   printf("---- segments %i\n",numSegments);
    for (int segID = 0; segID < numSegments; segID++) {
      LinearSegment segment;
      bool segmentStillValid
        = clipSegment(segment,segments[segID],hit_t,dbg);

      if (!segmentStillValid)
        continue;

      if (segment.end.t > ray.tMax) {
        // if (dbg)
        //   printf("INVALID SEGMENT segend %f hit_t %f ray.tmax %f\n",
        //          segment.end.t,hit_t,ray.tMax);
      }
      // if (dbg)
      //   printf(" seg %i clipped w/ scalar (%f %f)(%f %f)\n",
      //          segID,
      //          segment.begin.t,
      //          segment.begin.scalar,
      //          segment.end.t,
      //          segment.end.scalar);
      
      // compute a majorant for this segment
      range1f scalarRange = segment.scalarRange(); 
      float majorant
        = self.xf.majorant(scalarRange);
      // if (dbg)
      //   printf(" seg scalar range %f %f majorant %f\n",
      //          scalarRange.lower,
      //          scalarRange.upper,
      //          majorant);
      if (majorant == 0.f)
        continue;

      // we have a valid segment, with actual, non-zero
      // majorant... -> woodcock
      float t = segment.begin.t;
      // if (dbg)
      //   printf("## woodcock on segment ....\n");
      while (true) {
        // take a step...
        float dt = - logf(1.f-rand())/majorant;
        t += dt;
        // if (dbg) printf(" dt %f new t %f\n",dt,t);
        if (t >= min(hit_t,segment.end.t))
          // if (t >= min(segment.end.t,ray.tMax))
          break;
            
        // compute scalar by linearly interpolating along segment
        const float scalar = segment.lerp(t);
        vec4f mapped = self.xf.map(scalar);

        // if (dbg)
        //   printf(" -> at %f: scalar %f mapped (%f %f %f : %f)\n",
        //          t,scalar,mapped.x,mapped.y,mapped.z,mapped.w);
        // now got a scalar: compare to majorant
        float r = rand();
        // if (dbg)
        //   printf("   sampling change %f mapped %f majorant %f\n",
        //          r,mapped.w,majorant);
        bool accept = (mapped.w >= r*majorant);
        if (!accept) 
          continue;

        // if (dbg) printf("$$$$ HIT at %f\n",t);
        
        // we DID have a hit here!
        hit_t = t;
        // ray.tMax = t;
        inputLeafRange.upper = min(inputLeafRange.upper,hit_t);
        vec3f P = ray.org + t * ray.dir;
        ray.setVolumeHit(P,t,getPos(mapped));
        break;
      }
    }
    numSegments = 0;
  }

  /*! do all non-tet-non-grid elemnets in current leaf, using
    specialized ElementIntersector helper class */
  inline __device__ void NewIntersector::doOthers()
  {
    ElementIntersector isec(self,ray,inputLeafRange);
    int it = begin;
    while (it < end) {
      // find next prim:
      int next = it++;
      Element elt = self.elements[next];
      if (// elt.type == Element::GRID ||
          elt.type == Element::TET)
        continue;
      isec.setElement(elt);
      if (!isec.computeElementRange())
        continue;
        
      // compute majorant for given overlap range
      float majorant = isec.computeRangeMajorant();
      if (majorant == 0.f)
        continue;
        
      float t = isec.elementTRange.lower;
      while (true) {
        float dt = - logf(1-rand())/(majorant);
        t += dt;
        if (t >= isec.elementTRange.upper)
          break;
          
        vec3f P = ray.org + t * ray.dir;
        isec.sampleAndMap(P,ray.dbg);
        float r = rand();
        bool accept = (isec.mapped.w >= r*majorant);
        if (!accept) 
          continue;

        isec.leafRange.upper = hit_t;
        hit_t = t;
        ray.setVolumeHit(P,t,getPos(isec.mapped));
        break;
      }
    } 
  }

  /*! sample a gridlet on given position */
  inline __device__
  float NewIntersector::sampleGrid(const box4f &domain,
                                   vec3i dims,
                                   const float *scalars,
                                   vec3f P)
  {
    const box3f bounds = box3f((const vec3f &)domain.lower,
                               (const vec3f &)domain.upper);
      
    if (!bounds.contains(P))
      return NAN;

    vec3i numScalars = dims+1;
    vec3f cellSize = bounds.size()/vec3f(dims);
    vec3f objPos = (P-bounds.lower)/cellSize;
    vec3i imin(objPos);
    vec3i imax = min(imin+1,numScalars-1);

    auto linearIndex = [numScalars](const int x, const int y, const int z) {
      return z*numScalars.y*numScalars.x + y*numScalars.x + x;
    };

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
      return NAN;//false;
      
    vec3f frac = objPos-vec3f(imin);
      
    float f12 = lerp(f1,f2,frac.x);
    float f56 = lerp(f5,f6,frac.x);
    float f34 = lerp(f3,f4,frac.x);
    float f78 = lerp(f7,f8,frac.x);
      
    float f1234 = lerp(f12,f34,frac.y);
    float f5678 = lerp(f56,f78,frac.y);
      
    float retVal = lerp(f1234,f5678,frac.z);
    return retVal;
  }

  /* do ray-gridlet element intersection; currently doing only
     woodcock marching directly on that gridlets, using the entire
     gridlet's majorant. TODO: at some point compute how many
     samples to expect for full gridlet, and if too high, swtich to
     DDA and per-cell intersection */
  // inline __device__
  // void NewIntersector::doGrid(int gridID)
  // {
  //   box4f domain = self.gridDomains[gridID];
  //   vec3i dims = self.gridDims[gridID];
  //   range1f gridRange = inputLeafRange;
  //   if (!boxTest(gridRange.lower,gridRange.upper,
  //                        domain,ray.org,ray.dir))
  //     return;

  //   float majorant = self.xf.majorant(getRange(domain));
  //   if (majorant == 0.f)
  //     return;

  //   // TODO: compute expected num steps, and if too high, do DDA
  //   // across cells
  //   const float *scalars = self.gridScalars +
  //     self.gridOffsets[gridID];

  //   float t = gridRange.lower;
  //   while (true) {
  //     // take a step...
  //     float dt = - logf(1.f-rand())/majorant;
  //     t += dt;
  //     if (t >= gridRange.upper)
  //       break;
        
  //     // compute scalar by linearly interpolating along segment
  //     vec3f P = ray.org + t*ray.dir;
  //     const float scalar = sampleGrid(domain,dims,scalars,P);
  //     if (isnan(scalar)) 
  //       continue;
        
  //     vec4f mapped = self.xf.map(scalar);
            
  //     // now got a scalar: compare to majorant
  //     float r = rand();
  //     bool accept = (mapped.w >= r*majorant);
  //     if (!accept) 
  //       continue;
        
  //     // we DID have a hit here!
  //     hit_t = t;
  //     // ray.tMax = t;
  //     inputLeafRange.upper = min(inputLeafRange.upper,hit_t);
  //     ray.setVolumeHit(P,t,getPos(mapped));
  //     break;
  //   }
  // }
    
  // /*! intesect all gridlets in currelt leaf */
  // inline __device__ void NewIntersector::doGrids()
  // {
  //   for (int it=begin;it<end;it++) {
  //     const Element elt = self.elements[it];
  //     if (elt.type != Element::GRID)
  //       continue;
  //     doGrid(elt.ID);
  //   }
  // }
    
  /*! intesect all test in currelt leaf, by gathering segments and
    intersecting those when required */
  inline __device__ void NewIntersector::doTets()
  {
    TetIntersector isec;
    int it = begin;
    // if (dbg) printf("doing tets %i...%i\n",begin,end);
    while (it < end) {
      // ------------------------------------------------------------------
      // make list of segments that do have a geometric overlap
      // ------------------------------------------------------------------
      while (it < end) {
        // go to next element, and check if it's a tet.
        const Element elt = self.elements[it++];
        const bool hadATet 
          = isec.set(self,elt);
        if (!hadATet) {
          // if (elt.type == Element::GRID)
          //   hadAnyGrids = true;
          continue;
        }

        

        // it is a tet - compute geometric overlap segment
        LinearSegment segment;
        const bool hadGeometricOverlap
          = isec.computeSegment(segment,ray,inputLeafRange);

        // if (dbg)
        //   printf("geom overlap (%f %f)\n",
        //          segment.begin.t,segment.end.t);
        
        if (!hadGeometricOverlap)
          continue;

        // we did have geometric overlap, save this for later..
        segments[numSegments++] = segment;

        // check if we used up all our cached segments space; and if
        // so, exit for now (to first consume the existing segments)
        // and let's come back later.
        if (numSegments == maxSegments) break;
      }

      doSegments();
    }
  }

  /*! the main function provided by this header file - intersecting a
    ray against a given leaf node in a object-space umesh bvh */
  inline __device__
  float intersectLeaf(Ray &ray,
                      range1f &inputLeafRange,
                      const UMeshObjectSpace::DD &self,
                      int begin,
                      int end,
                      bool dbg)
  {
    return NewIntersector(ray,inputLeafRange,self,begin,end,dbg).hit_t;
  }
}
