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
  
  struct ElementSampler{
    inline __device__ ElementSampler(const UMeshField::DD &dd,
                                     const TransferFunction::DD &xf)
      : dd(dd), xf(xf){}

    inline __device__ bool evalElement(Element el, const vec3f& P, float& scalar){
      return dd.eltScalar(scalar, el, P);
    }

    inline __device__ bool sampleAndMap(Element el, const vec3f& P, vec4f& mapped){
      float scalar;
      if (!evalElement(el, P, scalar)){
        mapped.w = -INFINITY;
        return false;
      }

      mapped = xf.map(scalar);
      return true;
    }

    inline __device__ bool sampleAndMapDensity(Element el, const vec3f& P, float& mappedDensity){
      float scalar;
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

  struct CentralDifference {
    inline __device__
    CentralDifference(const UMeshField::DD &dd,
                      const TransferFunction::DD &xf,
                      vec3f P, int begin, int end, bool dbg)
      : dd(dd), xf(xf), dbg(dbg)
    {
      const float delta = .1f;
      this->P[0] = P - delta * vec3f(1.f,0.f,0.f);
      this->P[1] = P + delta * vec3f(1.f,0.f,0.f);
      this->P[2] = P - delta * vec3f(0.f,1.f,0.f);
      this->P[3] = P + delta * vec3f(0.f,1.f,0.f);
      this->P[4] = P - delta * vec3f(0.f,0.f,1.f);
      this->P[5] = P + delta * vec3f(0.f,0.f,1.f);
      this->P[6] = P;

#pragma unroll
      for (int i=0;i<7;i++)
        density[i] = -INFINITY;

      eval(begin,end);
    }

    inline __device__
    void eval(int begin, int end){
      ElementSampler elsa(dd, xf);
      for (int ei=begin;ei<end;ei++) {
        const Element elt = dd.elements[ei];
        vec4f color;
        if (elsa.sampleAndMap(elt, P[6], color)){
          mappedColor.x = color.x;
          mappedColor.y = color.y;
          mappedColor.z = color.z;
          density[6] = color.w;
        }

        for (int i = 0; i < 6; ++i){
          float d;
          if (elsa.sampleAndMapDensity(elt, P[i], d)){
            density[i] = d;
          }
        }
      }

      if (density[6] < 0)
        {
          printf("NO HIT IN THE CENTERRRR?????????\n");
          N = vec3f(1.f);
          return;
        }

      for (int i = 0; i < 6; ++i) {
        if (density[i] < 0) {
          density[i] = density[6];
          P[i] = P[6];
        }
      }

      N.x = safeDiv(density[1]-density[0], P[1].x-P[0].x);
      N.y = safeDiv(density[3]-density[2], P[3].y-P[2].y);
      N.z = safeDiv(density[5]-density[4], P[5].z-P[4].z);
    }

    vec3f P[7];
    float density[7];
    vec3f N;
    const bool dbg;
    vec3f mappedColor;
    const UMeshField::DD &dd;
    const TransferFunction::DD &xf;
  };
  
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
      // if (dbg) printf("setting elemnet tpe %i\n",elt.type);
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
      vec3f N = cross(getPos(b)-getPos(a),getPos(c)-getPos(a));
      return dot(P-getPos(a),N);
    }

    // using inward-facing planes here, like vtk
    inline __device__
    void clipRangeToPlane(vec4f a, vec4f b, vec4f c, bool dbg = false)
    {
      vec3f N = cross(getPos(b)-getPos(a),getPos(c)-getPos(a));
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
          //scalar_t0 = evalPyr(ray.org+elementTRange.lower*ray.dir);
          //scalar_t1 = evalPyr(ray.org+elementTRange.upper*ray.dir);
          return {scalar_t0, scalar_t1};
        case Element::WED:
          scalar_t0 = min(v0.w, min(v1.w, min(v2.w, min(v3.w, min(v4.w, v5.w)))));
          scalar_t1 = max(v0.w, max(v1.w, max(v2.w, max(v3.w, max(v4.w, v5.w)))));
          //        scalar_t0 = evalWed(ray.org+elementTRange.lower*ray.dir);
          //        scalar_t1 = evalWed(ray.org+elementTRange.upper*ray.dir);
          return {scalar_t0, scalar_t1};
        case Element::HEX:
          scalar_t0 = min(v0.w, min(v1.w, min(v2.w, min(v3.w,
                                                        min(v4.w, min(v5.w, min(v6.w, v7.w)))))));

          scalar_t1 = max(v0.w, max(v1.w, max(v2.w, max(v3.w,
                                                        max(v4.w, max(v5.w, max(v6.w, v7.w)))))));
          //        scalar_t0 = evalHex(ray.org+elementTRange.lower*ray.dir);
          //        scalar_t1 = evalHex(ray.org+elementTRange.upper*ray.dir);
          return {scalar_t0, scalar_t1};
        }
      return {NAN, NAN};
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

}
