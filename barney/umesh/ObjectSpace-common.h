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

#include "barney/umesh/UMeshField.h"
#include <cuBQL/bvh.h>

namespace barney {

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
  
  struct UMeshObjectSpace {
    struct DD {
      TransferFunction::DD xf;
      UMeshField::DD       mesh;
    };
  };

#if 0
  struct ElementSampler{
    inline __device__ ElementSampler(const UMeshField::DD &dd,
                                     const TransferFunction::DD &xf)
      : dd(dd), xf(xf){}

    inline __device__
    bool evalElement(Element el, const vec3f &P, float &scalar)
    {
      return dd.eltScalar(scalar, el, P);
    }

    inline __device__
    bool sampleAndMap(Element el, const vec3f& P, vec4f& mapped)
    {
      float scalar = 0.f;
      if (!evalElement(el, P, scalar)){
        mapped.w = -INFINITY;
        return false;
      }

      mapped = xf.map(scalar);
      return true;
    }

    inline __device__
    bool sampleAndMapDensity(Element el, const vec3f& P, float &mappedDensity)
    {
      float scalar = 0.f;
      if (!evalElement(el, P, scalar)){
        mappedDensity = -INFINITY;
        return false;
      }

      mappedDensity = xf.map(scalar).w;
      return true;
    }

    const UMeshField::DD &dd;
    const TransferFunction::DD &xf;
  };
#endif

  struct LinearSegment {
    struct EndPoint { float t, scalar; };

    inline __device__
    range1f scalarRange() const
    { return { min(begin.scalar,end.scalar),max(begin.scalar,end.scalar) }; }
    
    EndPoint begin, end;
  };

  /*! this code assumes that t is INSIDE that segment! */
  inline __device__
  float lerpScalar(const LinearSegment &segment, float t)
  {
    const float len = segment.end.t-segment.begin.t;
    if (len == 0.f) return segment.begin.scalar;
    
    const float f = (t-segment.begin.t)/len;
    return 
      (1.f-f)*segment.begin.scalar
      +    f *segment.end.scalar;
  }

  inline __device__
  bool clipSegment(LinearSegment &clipped,
                   const LinearSegment &original,
                   float t)
  {
    if (t < original.begin.t) return false;
    
    clipped = original;
    if (t >= clipped.end.t) {
      // nothing to do
    } else {
      clipped.end.scalar = lerpScalar(original,t);
      clipped.end.t = t;
    }
    return clipped.end.t > clipped.begin.t;
  }
  
  /*! helper class that represents a plane equation defiend through
      three points; can be used to clip a ray segment or evaluate
      distance to a point */
  struct Plane {
    inline __device__ void set(vec3f a, vec3f b, vec3f c)
    { N = cross(b-a,c-a); P = a; }
    
    inline __device__ bool clip(LinearSegment &segment,
                                const Ray &ray) const;

    inline __device__ float eval(vec3f v) const
    { return dot(v-P,N); }
    
    vec3f N;
    vec3f P;
  };

  inline __device__
  bool Plane::clip(LinearSegment &segment,
                   const Ray &ray)
    const
  {
    float NdotD = dot(N,ray.dir);
                    
    if (NdotD == 0.f) { //segment = range1f(); return false; }
      if (eval(ray.org) <= 0.f) {
        segment.begin.t = INFINITY;
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
  
  struct TetIntersector {
    inline __device__
    bool set(const UMeshObjectSpace::DD &dd,
             Element elt);
    inline __device__
    void set(float4 v0, float4 v1, float4 v2, float4 v3);
    
    inline __device__
    bool computeSegment(LinearSegment &segment,
                        const Ray &ray,
                        const range1f &inputRange) const;

    inline __device__
    float lerp_inside(const vec3f P) const;
    
    float4 _v0, _v1, _v2, _v3;
    vec3f v0, v1, v2, v3;
    Plane p0, p1, p2, p3;
  };
  

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
    
    return (t0*_v0.w + t1*_v1.w + t2*_v2.w + t3*_v3.w)
      / (t0+t1+t2+t3);
  }
  
  inline __device__
  void TetIntersector::set(float4 __v0,
                           float4 __v1,
                           float4 __v2,
                           float4 __v3)
  {
    _v0 = __v0;
    _v1 = __v1;
    _v2 = __v2;
    _v3 = __v3;
    
    v0 = getPos(_v0);
    v1 = getPos(_v1);
    v2 = getPos(_v2);
    v3 = getPos(_v3);
    
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

    int4 indices = dd.mesh.tetIndices[elt.ID];
    
    _v0 = dd.mesh.vertices[indices.x];
    _v1 = dd.mesh.vertices[indices.y];
    _v2 = dd.mesh.vertices[indices.z];
    _v3 = dd.mesh.vertices[indices.w];
    
    v0 = getPos(_v0);
    v1 = getPos(_v1);
    v2 = getPos(_v2);
    v3 = getPos(_v3);
    
    p3.set(v0,v1,v2);
    p2.set(v0,v3,v1);
    p1.set(v0,v2,v3);
    p0.set(v1,v3,v2);

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
  
  struct ElementIntersector
  {
    inline __device__
    ElementIntersector(const UMeshObjectSpace::DD &dd,
                       Ray &ray,
                       range1f leafRange
                       )
      : dd(dd), ray(ray), leafRange(leafRange), dbg(ray.dbg)
    {}

    const bool dbg;
    
    inline __device__
    void sampleAndMap(vec3f P, bool dbg);

    inline __device__
    bool setElement(Element elt)
    {
      element = elt;
      switch (elt.type)
        {
        case Element::TET: {
          int4 indices = dd.mesh.tetIndices[elt.ID];
          v0 = dd.mesh.vertices[indices.x];
          v1 = dd.mesh.vertices[indices.y];
          v2 = dd.mesh.vertices[indices.z];
          v3 = dd.mesh.vertices[indices.w];
        }
          break;
        case Element::PYR: {
          auto pyrIndices = dd.mesh.pyrIndices[elt.ID];
          v0 = dd.mesh.vertices[pyrIndices[0]];
          v1 = dd.mesh.vertices[pyrIndices[1]];
          v2 = dd.mesh.vertices[pyrIndices[2]];
          v3 = dd.mesh.vertices[pyrIndices[3]];
          v4 = dd.mesh.vertices[pyrIndices[4]];
        }
          break;
        case Element::WED: {
          auto wedIndices = dd.mesh.wedIndices[elt.ID];
          v0 = dd.mesh.vertices[wedIndices[0]];
          v1 = dd.mesh.vertices[wedIndices[1]];
          v2 = dd.mesh.vertices[wedIndices[2]];
          v3 = dd.mesh.vertices[wedIndices[3]];
          v4 = dd.mesh.vertices[wedIndices[4]];
          v5 = dd.mesh.vertices[wedIndices[5]];
        }
          break;
        case Element::HEX: {
          auto hexIndices = dd.mesh.hexIndices[elt.ID];
          v0 = dd.mesh.vertices[hexIndices[0]];
          v1 = dd.mesh.vertices[hexIndices[1]];
          v2 = dd.mesh.vertices[hexIndices[2]];
          v3 = dd.mesh.vertices[hexIndices[3]];
          v4 = dd.mesh.vertices[hexIndices[4]];
          v5 = dd.mesh.vertices[hexIndices[5]];
          v6 = dd.mesh.vertices[hexIndices[6]];
          v7 = dd.mesh.vertices[hexIndices[7]];
        }
          break;
        default:
          return false;
        }
      return true;
    }

    // using inward-facing planes here, like vtk
    inline __device__
    float evalToPlane(vec3f P, vec4f a, vec4f b, vec4f c)
    {
      vec3f N = (cross(getPos(b)-getPos(a),getPos(c)-getPos(a)));
      return dot(P-getPos(a),N);
    }

    // using inward-facing planes here, like vtk
    inline __device__
    void clipRangeToPlane(vec4f a, vec4f b, vec4f c, bool dbg = false)
    {
      vec3f N = (cross(getPos(b)-getPos(a),getPos(c)-getPos(a)));
      float NdotD = dot((vec3f)ray.dir, N);
      // if (dbg)
      //   printf(" clipping: N is %f %f %f, NdotD %f\n",
      //          N.x,N.y,N.z,NdotD);
      if (NdotD == 0.f)
        return;
      float plane_t = dot(getPos(a) - ray.org, N) / NdotD;
      if (NdotD < 0.f)
        elementTRange.upper = min(elementTRange.upper,plane_t);
      else
        elementTRange.lower = max(elementTRange.lower,plane_t);

      // if (dbg)
      //   printf(" clipping to t_plane %f, range now %f %f\n",
      //          plane_t, elementTRange.lower, elementTRange.upper);
    }

    inline __device__
    void clipRangeToPatch(vec4f a, vec4f b, vec4f c, vec4f d, bool dbg = false)
    {
      vec3f NRef = cross(getPos(b)-getPos(a),getPos(c)-getPos(a));
      vec3f ad = getPos(d) - getPos(a);

      if (dot(NRef, ad) >= 0){
        // abc - acd
        clipRangeToPlane(a, b, c, dbg);
        clipRangeToPlane(a, c, d, dbg);
      } else {
        // abd - bcd
        clipRangeToPlane(a, b, d, dbg);
        clipRangeToPlane(b, c, d, dbg);
      }
    }

    inline __device__ bool evalElement(const vec3f& P, float& sample) {
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

    inline __device__
    bool evalTet(vec3f P, float& sample, bool dbg = false)
    {
#if 1
      float t3 = evalToImplicitPlane(P,v0,v1,v2,v3);
      if (t3 < 0.f) return false;
      float t2 = evalToImplicitPlane(P,v0,v3,v1,v2);
      if (t2 < 0.f) return false;
      float t1 = evalToImplicitPlane(P,v0,v2,v3,v1);
      if (t1 < 0.f) return false;
      float t0 = evalToImplicitPlane(P,v1,v3,v2,v0);
      if (t0 < 0.f) return false;
#else
      float t3 = evalToImplicitPlane(P,v0,v1,v2);
      if (t3 < 0.f) return false;
      float t2 = evalToImplicitPlane(P,v0,v3,v1);
      if (t2 < 0.f) return false;
      float t1 = evalToImplicitPlane(P,v0,v2,v3);
      if (t1 < 0.f) return false;
      float t0 = evalToImplicitPlane(P,v1,v3,v2);
      if (t0 < 0.f) return false;
#endif
      float scale = 1.f/(t0+t1+t2+t3);
      sample = scale * (t0*v0.w + t1*v1.w + t2*v2.w + t3*v3.w);
      return true;
    }
    //======================================================================//

    inline __device__
    bool evalPyr(vec3f P, float& sample, bool dbg = false)
    {
      return intersectPyrEXT(sample, P, v0, v1, v2, v3, v4);
    }

    //======================================================================//
    inline __device__
    bool evalWed(vec3f P, float& sample, bool dbg = false)
    {
      return intersectWedgeEXT(sample, P, v0, v1, v2, v3, v4, v5);
    }

    //======================================================================//
    inline __device__
    bool evalHex(vec3f P, float& sample, bool dbg = false)
    {
      return intersectHexEXT(sample, P, v0, v1, v2, v3, v4, v5, v6, v7);
    }

    inline __device__
    range1f computeElementScalarRange() 
    {
      float scalar_t0 = 0.f;
      float scalar_t1 = 0.f;
      switch (element.type)
        {
        case Element::TET:
          evalTet(ray.org+elementTRange.lower*ray.dir, scalar_t0);
          evalTet(ray.org+elementTRange.upper*ray.dir, scalar_t1);
          return { min(scalar_t0,scalar_t1),max(scalar_t0,scalar_t1) };
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
    
    inline __device__
    bool computeElementRange(bool dbg = false)
    {
      elementTRange = leafRange;
      // if (dbg) printf("eltrange at start %f %f\n",
      //                 elementTRange.lower,
      //                 elementTRange.upper);
      switch (element.type)
        {
        case Element::TET:
          clipRangeToPlane(v0,v1,v2,dbg);
          clipRangeToPlane(v0,v3,v1,dbg);
          clipRangeToPlane(v0,v2,v3,dbg);
          clipRangeToPlane(v1,v3,v2,dbg);
          break;
        case Element::PYR:
          clipRangeToPlane(v0, v4, v1);
          clipRangeToPlane(v0, v3, v4);
          clipRangeToPlane(v1, v4, v2);
          clipRangeToPlane(v2, v4, v3);
          clipRangeToPatch(v0, v1, v2, v3);
          break;
        case Element::WED:
          clipRangeToPlane(v0,v2,v1,dbg);
          clipRangeToPlane(v3,v4,v5,dbg);
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

    inline __device__
    float computeRangeMajorant()
    {
      return dd.xf.majorant(computeElementScalarRange());
    }
    
    Ray &ray;

    // current t positoin's scalar and mapped avlues
    float scalar;
    vec4f mapped;
    
    // parameter interval of where ray has valid overlap with the
    // leaf's bbox (ie, where ray overlaps the box, up to tMax/tHit
    range1f leafRange;
    
    // paramter interval where ray's valid range overlaps the current *element*
    range1f elementTRange;
    
    // current tet:
    Element element;
    vec4f v0, v1, v2, v3,
      v4, v5, v6, v7;

    const UMeshObjectSpace::DD &dd;
  };
  
  inline __device__
  void ElementIntersector::sampleAndMap(vec3f P, bool dbg)
  {
    if (!evalElement(P, scalar))
      mapped = 0.f;
    else
      mapped = dd.xf.map(scalar,dbg);
  }

  inline __device__
  bool boxTest(float &t0, float &t1,
               box3f box,
               const vec3f org,
               const vec3f dir)
  {
    vec3f t_lo = (box.lower - org) * rcp(dir);
    vec3f t_hi = (box.upper - org) * rcp(dir);
    vec3f t_nr = min(t_lo,t_hi);
    vec3f t_fr = max(t_lo,t_hi);
    t0 = max(t0,reduce_max(t_nr));
    t1 = min(t1,reduce_min(t_fr));
    return t0 < t1;
  }

  inline __device__
  bool boxTest(float &t0, float &t1,
               box4f box,
               const vec3f org,
               const vec3f dir)
  {
    return boxTest(t0,t1,box3f({box.lower.x,box.lower.y,box.lower.z},
                               {box.upper.x,box.upper.y,box.upper.z}),
                   org,dir);
  }

#if 1

  struct NewIntersector {
    enum { maxSegments = 4 };

    inline __device__
    NewIntersector(Ray &ray,
                           range1f &inputLeafRange,
                           const UMeshObjectSpace::DD &self,
                           int begin,
                           int end)
      : ray(ray), self(self), inputLeafRange(inputLeafRange),
        begin(begin), end(end), rand((LCG<4> &)ray.rngSeed)

    {
      doTets();
      if (hadAnyGrids)
        doGrids();
      if (hadAnyOthers)
        doOthers();
    }

    inline __device__ void doSegments()
    {
      for (int segID = 0; segID < numSegments; segID++) {
        LinearSegment segment;
        bool segmentStillValid
          = clipSegment(segment,segments[segID],hit_t);
        if (!segmentStillValid)
          continue;

        // compute a majorant for this segment
        float majorant
          = self.xf.majorant(segment.scalarRange());
        if (majorant == 0.f)
          continue;

        // we have a valid segment, with actual, non-zero
        // majorant... -> woodcock
        float t = segment.begin.t;
        while (true) {
          // take a step...
          float dt = - logf(1.f-rand())/majorant;
          t += dt;
          if (t >= segment.end.t)
            // if (t >= min(segment.end.t,ray.tMax))
            break;
            
          // compute scalar by linearly interpolating along segment
          const float scalar = lerpScalar(segment,t);
          vec4f mapped = self.xf.map(scalar);
            
          // now got a scalar: compare to majorant
          float r = rand();
          bool accept = (mapped.w >= r*majorant);
          if (!accept) 
            continue;

          // we DID have a hit here!
          hit_t = t;
          // ray.tMax = t;
          inputLeafRange.upper = min(inputLeafRange.upper,hit_t);
          ray.hit.baseColor = getPos(mapped);
          break;
        }
      }
      numSegments = 0;
    }

    inline __device__
    float4 average(float4 a,float4 b,float4 c,float4 d)
    {
      return make_float4(.25f*(a.x+b.x+c.x+d.x),
                         .25f*(a.y+b.y+c.y+d.y),
                         .25f*(a.z+b.z+c.z+d.z),
                         .25f*(a.w+b.w+c.w+d.w));
    }

    inline __device__
    float4 average(float4 a,float4 b,float4 c,float4 d,
                   float4 e,float4 f,float4 g,float4 h)
    {
      return make_float4(.125f*(a.x+b.x+c.x+d.x + e.x+f.x+g.x+h.x),
                         .125f*(a.y+b.y+c.y+d.y + e.x+f.x+g.x+h.x),
                         .125f*(a.z+b.z+c.z+d.z + e.x+f.x+g.x+h.x),
                         .125f*(a.w+b.w+c.w+d.w + e.x+f.x+g.x+h.x));
    }
    
    // inline __device__ void doTet(float4 v0,
    //                              float4 v1,
    //                              float4 v2,
    //                              float4 v3)
    // {
    //   // if ((const vec3f&)v0 == (const vec3f&)v1
    //   //     || (const vec3f&)v0 == (const vec3f&)v2
    //   //     || (const vec3f&)v0 == (const vec3f&)v3
    //   //     || (const vec3f&)v1 == (const vec3f&)v2
    //   //     || (const vec3f&)v1 == (const vec3f&)v3
    //   //     || (const vec3f&)v2 == (const vec3f&)v3)
    //     // return;
    //   isec.set(v0,v1,v2,v3);
    //   if (isec.p0.eval(isec.v0) <= 0.f)
    //     return;
    //   // it is a tet - compute geometric overlap segment
    //   LinearSegment segment;
    //   const bool hadGeometricOverlap
    //     = isec.computeSegment(segment,ray,inputLeafRange);
    //   if (!hadGeometricOverlap)
    //     return;

    //   // we did have geometric overlap, save this for later..
    //   segments[numSegments++] = segment;
    // }
    
    // inline __device__ void doPyrSegs(float4 v0,
    //                                  float4 v1,
    //                                  float4 v2,
    //                                  float4 v3,
    //                                  float4 v4)
    // {
    //   float4 center = average(v0,v1,v2,v3);
    //   doTet(v0,v1,center,v4);
    //   doTet(v1,v2,center,v4);
    //   doTet(v2,v3,center,v4);
    //   doTet(v3,v0,center,v4);
    //   doSegments();
    // }
    
    inline __device__ void doOthers()
    {
      ElementIntersector isec(self,ray,inputLeafRange);
      int it = begin;
      while (it < end) {
        // find next prim:
        int next = it++;
        Element elt = self.mesh.elements[next];
        if (elt.type == Element::GRID || elt.type == Element::TET)
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
          bool accept = (isec.mapped.w > r*majorant);
          if (!accept) 
            continue;
          
          hit_t = t;
          isec.leafRange.upper = hit_t;
          ray.hit.baseColor = getPos(isec.mapped);
          break;
        }
      } 
    }
      
    inline __device__
    float sampleGrid(const box4f &domain, vec3i dims, const float *scalars,
                     vec3f P)
    {
#if 1
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

      // const float *scalars = gridScalars + gridOffsets[primID];

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
      
#else
      P = (P - getPos(domain.lower)) * rcp(getPos(domain.size())) * vec3f(dims);
      vec3i cell = max(vec3i(0),min(dims-1,vec3i(P)));
      vec3f frac = P - vec3f(cell);
#if 0
      int idx = cell.x + (dims.x+1) * (cell.y + (dims.y+1)*(cell.z));

      return scalars[idx];
#else
      
      // int sx = dims.x+1;
      // int sy = dims.y+1;

      // int ix0 = cell.x;
      // int iy0 = cell.y;
      // int iz0 = cell.z;
      // int ix1 = ix0+1;
      // int iy1 = iy0+1;
      // int iz1 = iz0+1;

      // int i000 = ix0+sx*(iy0+sy*(iz0));
      // int i001 = ix1+sx*(iy0+sy*(iz0));
      // int i010 = ix0+sx*(iy1+sy*(iz0));
      // int i011 = ix1+sx*(iy1+sy*(iz0));

      // int i100 = ix0+sx*(iy0+sy*(iz1));
      // int i101 = ix1+sx*(iy0+sy*(iz1));
      // int i110 = ix0+sx*(iy1+sy*(iz1));
      // int i111 = ix1+sx*(iy1+sy*(iz1));

      // float f000 = scalars[i000];
      // float f001 = scalars[i001];
      // float f010 = scalars[i010];
      // float f011 = scalars[i011];
      // float f100 = scalars[i100];
      // float f101 = scalars[i101];
      // float f110 = scalars[i110];
      // float f111 = scalars[i111];
      
      auto scalar = [scalars,cell,dims](int ix, int iy, int iz)
      {
        return scalars[(cell.x+ix)+(dims.x+1)*((cell.y+iy)+(dims.y+1)*(cell.z+iz))];
      };
      float f000 = (1.f-frac.x)*(1.f-frac.y)*(1.f-frac.z)*scalar(0,0,0);
      float f001 = (    frac.x)*(1.f-frac.y)*(1.f-frac.z)*scalar(1,0,0);
      float f010 = (1.f-frac.x)*(    frac.y)*(1.f-frac.z)*scalar(0,1,0);
      float f011 = (    frac.x)*(    frac.y)*(1.f-frac.z)*scalar(1,1,0);
      float f100 = (1.f-frac.x)*(1.f-frac.y)*(    frac.z)*scalar(0,0,1);
      float f101 = (    frac.x)*(1.f-frac.y)*(    frac.z)*scalar(1,0,1);
      float f110 = (1.f-frac.x)*(    frac.y)*(    frac.z)*scalar(0,1,1);
      float f111 = (    frac.x)*(    frac.y)*(    frac.z)*scalar(1,1,1);
      return (f000+f001+f010+f011+f100+f101+f110+f111);
#endif
#endif
    }
    
    inline __device__
    void doGrid(int gridID)
    {
      box4f domain = self.mesh.gridDomains[gridID];
      vec3i dims = self.mesh.gridDims[gridID];
#if 0
      vec3f cellWidth = getPos(domain.size()) * rcp(vec3f(dims)+1.f);
      (vec3f&)domain.lower = (vec3f&)domain.lower + .5f * cellWidth;
      (vec3f&)domain.upper = (vec3f&)domain.upper - .5f * cellWidth;
#endif
      range1f gridRange = inputLeafRange;
      if (!boxTest(gridRange.lower,gridRange.upper,
                   domain,ray.org,ray.dir))
        return;

      float majorant = self.xf.majorant(getRange(domain));
      if (majorant == 0.f)
        return;

      // TODO: compute expected num steps, and if too high, do DDA
      // across cells
      const float *scalars = self.mesh.gridScalars +
        self.mesh.gridOffsets[gridID];

      float t = gridRange.lower;
      while (true) {
        // take a step...
        float dt = - logf(1.f-rand())/majorant;
        t += dt;
        if (t >= gridRange.upper)
          break;
        
        // compute scalar by linearly interpolating along segment
        vec3f P = ray.org + t*ray.dir;
        const float scalar = sampleGrid(domain,dims,scalars,P);
        if (isnan(scalar)) 
          continue;
        
        vec4f mapped = self.xf.map(scalar);
            
        // now got a scalar: compare to majorant
        float r = rand();
        bool accept = (mapped.w >= r*majorant);
        if (!accept) 
          continue;
        
        // we DID have a hit here!
        hit_t = t;
        // ray.tMax = t;
        inputLeafRange.upper = min(inputLeafRange.upper,hit_t);
        ray.hit.baseColor = getPos(mapped);
        break;
      }
    }
    
    inline __device__ void doGrids()
    {
      for (int it=begin;it<end;it++) {
        const Element elt = self.mesh.elements[it];
        if (elt.type != Element::GRID)
          continue;
        doGrid(elt.ID);
      }
    }
    
    inline __device__ void doTets()
    {
      TetIntersector isec;
      int it = begin;
      while (it < end) {
        // ------------------------------------------------------------------
        // make list of segments that do have a geometric overlap
        // ------------------------------------------------------------------
        while (it < end) {
          // go to next element, and check if it's a tet.
          const Element elt = self.mesh.elements[it++];
          const bool hadATet 
            = isec.set(self,elt);
          if (!hadATet) {
            if (elt.type == Element::GRID)
              hadAnyGrids = true;
            else
              hadAnyOthers = true;
            continue;
          }

          // it is a tet - compute geometric overlap segment
          LinearSegment segment;
          const bool hadGeometricOverlap
            = isec.computeSegment(segment,ray,inputLeafRange);
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

    bool hadAnyGrids = false;
    bool hadAnyOthers = false;
    int numSegments = 0;
    LinearSegment segments[maxSegments];
    const UMeshObjectSpace::DD self;
    Ray &ray;
    range1f &inputLeafRange;
    const int begin;
    const int end;
    float hit_t = INFINITY;
    LCG<4> &rand;
  };
  
  inline __device__
  float intersectLeaf(Ray &ray,
                      range1f &inputLeafRange,
                      const UMeshObjectSpace::DD &self,
                      int begin,
                      int end)
  {
#if 1
    return NewIntersector(ray,inputLeafRange,self,begin,end).hit_t;
#else
    const bool dbg = ray.dbg;
    LCG<4> &rand = (LCG<4> &)ray.rngSeed;
    
    enum { maxSegments = 4 };
    LinearSegment segments[maxSegments];
    bool haveSkippedAnyNonTetPrimitives = false;
    float hit_t = INFINITY;
    
    { // do all TETS
      TetIntersector isec;
      int it = begin;
      while (it < end) {
        int numSegments = 0;
        // ------------------------------------------------------------------
        // make list of segments that do have a geometric overlap
        // ------------------------------------------------------------------
        while (it < end) {
          // go to next element, and check if it's a tet.
          const bool hadATet
            = isec.set(self,self.mesh.elements[it++]);
          if (!hadATet) {
            haveSkippedAnyNonTetPrimitives = true;
            continue;
          }

          // it is a tet - compute geometric overlap segment
          LinearSegment segment;
          const bool hadGeometricOverlap
            = isec.computeSegment(segment,ray,inputLeafRange);
          if (!hadGeometricOverlap)
            continue;

          // we did have geometric overlap, save this for later..
          segments[numSegments++] = segment;

          // check if we used up all our cached segments space; and if
          // so, exit for now (to first consume the existing segments)
          // and let's come back later.
          if (numSegments == maxSegments) break;
        }

        // ------------------------------------------------------------------
        // at this point we either have (geometrically) intersected
        // all tets, or had to interrupt tet processing because we're
        // out of segment space. either way need to process whatever
        // segments we have.
        // ------------------------------------------------------------------
        for (int segID = 0; segID < numSegments; segID++) {
          LinearSegment segment;
          bool segmentStillValid
            = clipSegment(segment,segments[segID],hit_t);
          if (!segmentStillValid)
            continue;

          // compute a majorant for this segment
          float majorant
            = self.xf.majorant(segment.scalarRange());
          if (majorant == 0.f)
            continue;

          // we have a valid segment, with actual, non-zero
          // majorant... -> woodcock
          float t = segment.begin.t;
          while (true) {
            // take a step...
            float dt = - logf(1.f-rand())/majorant;
            t += dt;
            if (t >= segment.end.t)
            // if (t >= min(segment.end.t,ray.tMax))
              break;
            
            // compute scalar by linearly interpolating along segment
            const float scalar = lerpScalar(segment,t);
            vec4f mapped = self.xf.map(scalar);
            
            // now got a scalar: compare to majorant
            float r = rand();
            bool accept = (mapped.w >= r*majorant);
            if (!accept) 
              continue;

            // we DID have a hit here!
            hit_t = t;
            // ray.tMax = t;
            inputLeafRange.upper = min(inputLeafRange.upper,hit_t);
            ray.hit.baseColor = getPos(mapped);
            break;
          }
        }

        // ------------------------------------------------------------------
        // we've processed whatever segments we had - check if there's more
        // ------------------------------------------------------------------
        if (it == end) break;
      }
    }

    // ==================================================================
    // done with all tets - now check if there's other elements as well
    // ==================================================================
    {
    }
    
    // ==================================================================
    // done with all elements... return
    // ==================================================================
    return hit_t;
#endif
  }
#else
  inline __device__
  float intersectLeaf(Ray &ray,
                      range1f &inputLeafRange,
                      const UMeshObjectSpace::DD &self,
                      int begin,
                      int end)
  {
    bool dbg = ray.dbg;
    LCG<4> &rand = (LCG<4> &)ray.rngSeed;
    int numStepsTaken = 0, numSamplesTaken = 0, numRangesComputed = 0, numSamplesRejected = 0;
    ElementIntersector isec(self,ray,inputLeafRange);
    // use prim box only to find candidates (similar to ray tracing
    // for triangles), but each ray then intersects each prim
    // individually.

#if PRINT_BALLOT
    int numActive = __popc(__ballot(1));
    if (ray.dbg)
      printf("### leaf isec #%i on geom %lx, range %f %f, numActive = %i\n",
             ray.numLeavesThisRay++,
             (void *)&self,inputLeafRange.lower,inputLeafRange.upper,
             numActive);
#endif
    int it = begin;
    Element hit_elt;
    float hit_t = INFINITY;
    while (it < end) {
      // find next prim:
      int next = it++;
#if PRINT_BALLOT
      int numActive = __popc(__ballot(1));
      if (dbg) printf(" prim %i (#%i on ray) (numActive=%i)\n",next,
                      ray.numPrimsThisRay++,
                      numActive);
#endif      
      if (!isec.setElement(self.mesh.elements[next]))
        continue;
                      
      // if (dbg) printf("element %i\n",isec.element.ID);
      
      // check for GEOMETRIC overlap of ray and prim
      numRangesComputed++;
      if (!isec.computeElementRange())
        continue;

      // compute majorant for given overlap range
      float majorant = isec.computeRangeMajorant();
#if PRINT_BALLOT
      if (dbg) printf("   > ray intersects *geometry* of prim; element range %f %f majorant is %f\n",
                      isec.elementTRange.lower,
                      isec.elementTRange.upper,
                      majorant);
#endif
      if (majorant == 0.f)
        continue;
      
      float t = isec.elementTRange.lower;
      while (true) {
        float dt = - logf(1-rand())/(majorant);
        t += dt;
        numStepsTaken++;
#if PRINT_BALLOT
        if (ray.dbg) printf("  > step taken %f, new t %f / %f\n",
                            dt,t,isec.elementTRange.upper);
#endif
        if (t >= isec.elementTRange.upper)
          break;

        vec3f P = ray.org + t * ray.dir;
        numSamplesTaken++;
        isec.sampleAndMap(P,dbg);
#if PRINT_BALLOT
        if (ray.dbg) printf("  >> ACTUAL sample\n");
#endif
        // if (!isec.sampleAndMap()) {
        //   if (dbg) printf("COULD NOT SAMPLE TET t %f range %f %f!?!?!?!\n",t,
        //                   isec.elementTRange.lower,
        //                   isec.elementTRange.upper
        //                   );
        //   continue;
        // }
        float r = rand();
        bool accept = (isec.mapped.w > r*majorant);
        if (!accept) {
          // if (dbg)printf("REJECTED w = %f, rnd = %f, majorant %f\n",
          //                isec.mapped.w,r,majorant);
          numSamplesRejected++;
          continue;
        }
        
        hit_t = t;
        hit_elt = isec.element;
        isec.leafRange.upper = hit_t;

        ray.hit.baseColor = getPos(isec.mapped);
        
        
#if PRINT_BALLOT
        if (dbg) printf("**** ACCEPTED at t = %f, P = %f %f %f, tet ID %i\n",
                        t, P.x,P.y,P.z,isec.element.ID);
#endif
        break;
      }
      
    }
    // if (dbg)
    //   printf("num ranges computed %i steps taken %i samples taken %i rejected %i\n",
    //          numRangesComputed, numStepsTaken, numSamplesTaken,
    //          numSamplesRejected);
    return hit_t;
  }
#endif
  
}
