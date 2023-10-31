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

#include "barney/unstructured/UMeshQCSampler.h"
#include "owl/owl_device.h"

namespace barney {

#if 0
  template<typename T>
  inline __device__
  void swap(T &a, T &b) { T c = a; a = b; b = c; }

  inline __device__
  float safeDiv(float a, float b) { return (b==0.f)?0.f:(a/b); }
  
  inline __device__
  vec4f DeviceXF::map(float s, bool dbg) const
  {
    float f = (s-domain.lower)/domain.span();
    f = clamp(f,0.f,1.f);
    f *= (numValues-1);
    int idx0 = clamp(int(f),0,numValues-1);
    int idx1 = clamp(idx0+1,0,numValues-1);
    f -= idx0;
    vec4f v0 = (vec4f)values[idx0];
    vec4f v1 = (vec4f)values[idx1];
    if (dbg)
      printf("map() indices %i %i values %f %f %f %f : %f %f %f %f f %f base %f\n",
             idx0,idx1,
             v0.x,v0.y,v0.z,v0.w,
             v1.x,v1.y,v1.z,v1.w,
             f,baseDensity);
        
      
    vec4f r = (1.f-f)*v0+f*v1;
    r.w *= baseDensity;
    return r;
  }

  struct MeshSampler
  {
    inline __device__
    MeshSampler(const UMeshQC::DD &dd) : dd(dd) {}

    inline __device__ void operator=(const MeshSampler &other)
    {
      P = other.P;
      mapped = other.mapped;
      scalar = other.scalar;
    }

    // inline __device__
    // void computeGradient()
    // {
    // };
    inline __device__
    bool sample();
    inline __device__
    bool sampleAndMap();
    inline __device__
    bool sampleAndMap(int elts_begin, int elts_end);
      
    Element element;
    vec3f P;

    float scalar;
    /*! color- and alpha-mapped sample */
    vec4f mapped;
    // vec3f gradient;

    const UMeshQC::DD &dd;    
  };

  struct ElementIntersector : public MeshSampler
  {
    inline __device__
    ElementIntersector(const UMeshQC::DD &dd,
                       Ray &ray,
                       range1f leafRange
                       )
      : MeshSampler(dd), ray(ray), leafRange(leafRange)
    {}

    inline __device__
    void sampleAndMap(bool dbg);

    inline __device__
    void setElement(Element elt)
    {
      element = elt;
      // currently only supporting tets ....
      int4 indices = dd.mesh.tetIndices[elt.ID];
      v0 = dd.mesh.vertices[indices.x];
      v1 = dd.mesh.vertices[indices.y];
      v2 = dd.mesh.vertices[indices.z];
      v3 = dd.mesh.vertices[indices.w];
      if (dot(getPos(v1)-getPos(v0),cross(getPos(v2)-getPos(v0),getPos(v3)-getPos(v0))) < 0.f) {
        swap(v0,v1);
      }
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
      float NdotD = dot(ray.dir, N);
      if (dbg)
        printf(" clipping: N is %f %f %f, NdotD %f\n",
               N.x,N.y,N.z,NdotD);
      if (NdotD == 0.f)
        return;
      float plane_t = dot(getPos(a) - ray.org, N) / NdotD;
      if (NdotD < 0.f)
        elementTRange.upper = min(elementTRange.upper,plane_t);
      else
        elementTRange.lower = max(elementTRange.lower,plane_t);

      if (dbg)
        printf(" clipping to t_plane %f, range now %f %f\n",
               plane_t, elementTRange.lower, elementTRange.upper);
    }

    inline __device__
    float evalTet(vec3f P, bool dbg = false)
    {
      float t3 = evalToPlane(P,v0,v1,v2);
      float t2 = evalToPlane(P,v0,v3,v1);
      float t1 = evalToPlane(P,v0,v2,v3);
      float t0 = evalToPlane(P,v1,v3,v2);

      float scale = 1.f/(t0+t1+t2+t3);
      if (dbg) printf("eval %f %f %f %f check %f -> scale %f, vtx.w %f %f %f %f\n",
                      t0,t1,t2,t3,evalToPlane(getPos(v3),v0,v1,v2),scale,
                      v0.w,v1.w,v2.w,v3.w);
      return scale * (t0*v0.w + t1*v1.w + t2*v2.w + t3*v3.w);
    }

    inline __device__
    range1f computeElementScalarRange()
    {
      float scalar_t0 = evalTet(ray.org+elementTRange.lower*ray.dir);
      float scalar_t1 = evalTet(ray.org+elementTRange.upper*ray.dir);
      return { min(scalar_t0,scalar_t1),max(scalar_t0,scalar_t1) };
    }
    
    inline __device__
    bool computeElementRange(bool dbg = false)
    {
      elementTRange = leafRange;
      if (dbg) printf("eltrange at start %f %f\n",
                      elementTRange.lower,
                      elementTRange.upper);
      clipRangeToPlane(v0,v1,v2,dbg);
      clipRangeToPlane(v0,v3,v1,dbg);
      clipRangeToPlane(v0,v2,v3,dbg);
      clipRangeToPlane(v1,v3,v2,dbg);
      return !elementTRange.empty();
    }

    inline __device__
    float computeRangeMajorant()
    {
      return dd.xf.majorant(computeElementScalarRange());
    }
    
    Ray &ray;
    // parameter interval of where ray has valid overlap with the
    // leaf's bbox (ie, where ray overlaps the box, up to tMax/tHit
    range1f leafRange;
    
    // paramter interval where ray's valid range overlaps the current *element*
    range1f elementTRange;
    
    // current tet:
    vec4f v0, v1, v2, v3;
  };

  inline __device__
  float doPlane(vec3f P, vec3f a, vec3f b, vec3f c)
  {
    vec3f n = cross(b-a,c-a);
    return dot(P-a,n);
  }
  
  
  inline __device__
  bool evaluateTet(vec3f v0, float s0,
                   vec3f v1, float s1,
                   vec3f v2, float s2,
                   vec3f v3, float s3,
                   float &scalar,
                   vec3f P)
  {
    float a = doPlane(v3,v0,v1,v2);
    if (a == 0.f) return false;
    if (a < 0.f) {
      swap(v0,v1);
      swap(s0,s1);
      a = -a;
    }
    // clipPlane(v0,v1,v2);
    // // if (tRange.empty()) return;
    // clipPlane(v0,v3,v1);
    // // if (tRange.empty()) return;
    // clipPlane(v0,v2,v3);
    // // if (tRange.empty()) return;
    // clipPlane(v1,v3,v2);

    float w3 = doPlane(P, v0,v1,v2)/a; if (w3 < 0.f) return false;
    float w2 = doPlane(v3,v0,v1,P )/a; if (w2 < 0.f) return false;
    float w1 = doPlane(v3,v0,P ,v2)/a; if (w1 < 0.f) return false;
    float w0 = doPlane(v3,P ,v1,v2)/a; if (w0 < 0.f) return false;

    // float ww = w0+w1+w2+w3;
    // if (ww < .98 || ww > 1.02)
    //   printf("weird w %f\n",ww);
    // float w2 = doPlane(P,v0,v3,v1)/a; if (w2 < 0.f) return false;
    // float w1 = doPlane(P,v0,v2,v3)/a; if (w1 < 0.f) return false;
    // float w0 = doPlane(P,v1,v3,v2)/a; if (w0 < 0.f) return false;
    scalar = w0*s0 + w1*s1 + w2*s2 + w3*s3;
    return true;
  }

  inline __device__
  bool MeshSampler::sample() 
  {
    auto &mesh = dd.mesh;
    if (element.type == UMeshQC::TET) {
      auto indices = mesh.tetIndices[element.ID];
      vec4f a = mesh.vertices[indices.x];
      vec4f b = mesh.vertices[indices.y];
      vec4f c = mesh.vertices[indices.z];
      vec4f d = mesh.vertices[indices.w];
      return evaluateTet(getPos(a),a.w,
                         getPos(b),b.w,
                         getPos(c),c.w,
                         getPos(d),d.w,
                         scalar, P);
    }
    return false;
  }
  
  inline __device__
  void ElementIntersector::sampleAndMap(bool dbg) 
  {
    scalar = evalTet(P,dbg);
    mapped = dd.xf.map(scalar,dbg);
    if (dbg)
      printf("***** scalar %f mapped %f %f %f: %f\n",scalar,
             mapped.x,mapped.y,mapped.z,mapped.w);
  }
  

  
  inline __device__
  bool MeshSampler::sampleAndMap() 
  {
    auto &mesh = dd.mesh;
    if (element.type == UMeshQC::TET) {
      auto indices = mesh.tetIndices[element.ID];
      vec4f a = mesh.vertices[indices.x];
      vec4f b = mesh.vertices[indices.y];
      vec4f c = mesh.vertices[indices.z];
      vec4f d = mesh.vertices[indices.w];
      if (!evaluateTet(getPos(a),a.w,
                       getPos(b),b.w,
                       getPos(c),c.w,
                       getPos(d),d.w,
                       scalar, P))
        return false;
      mapped = dd.xf.map(scalar);
      // gradient
      //   = (mesh.xf.map(a.w).w-mapped.w)*(getPos(a)-P)
      //   + (mesh.xf.map(b.w).w-mapped.w)*(getPos(b)-P)
      //   + (mesh.xf.map(c.w).w-mapped.w)*(getPos(c)-P)
      //   + (mesh.xf.map(d.w).w-mapped.w)*(getPos(d)-P);
      
      return true;
    }
    return false;
  }
  

  inline __device__
  bool MeshSampler::sampleAndMap(int elts_begin, int elts_end) 
  {
    auto &mesh = dd.mesh;
    for (int i=elts_begin;i<elts_end;i++) {
      element = mesh.elements[i];
      if (sampleAndMap())
        return true;
    }
    return false;
  }


  
      
  
  inline __device__
  float DeviceXF::majorant(range1f r, bool dbg) const
  {
    float f_lo = (r.lower-domain.lower)/domain.span();
    float f_hi = (r.upper-domain.lower)/domain.span();
    f_lo = clamp(f_lo,0.f,1.f);
    f_hi = clamp(f_hi,0.f,1.f);
    f_lo *= (numValues-1);
    f_hi *= (numValues-1);
    int idx0 = clamp(int(f_lo),0,numValues-1);
    int idx1 = clamp(int(f_hi)+1,0,numValues-1);
    float m = 0.f;
    for (int i=idx0;i<=idx1;i++)
      m = max(m,values[i].w);
    // printf("maj [%f %f] domain [%f %f]-> idx [%i %i] max %f dens %f\n",
    //        r.lower,r.upper,domain.lower,domain.upper,idx0,idx1,m,baseDensity);
    return m * baseDensity;
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
  
  OPTIX_BOUNDS_PROGRAM(UMeshQCBounds)(const void *geomData,                
                                      owl::common::box3f &bounds,  
                                      const int32_t primID)
  {
    const auto &self = *(const UMeshQC::DD *)geomData;

    box4f clusterBounds;
#if CLUSTERS_FROM_QC
    int begin = primID * UMeshQC::clusterSize;
    int end   = min(begin+UMeshQC::clusterSize,self.mesh.numElements);
#else
    int begin = self.clusters[primID].begin;
    int end   = self.clusters[primID].end;
#endif

    for (int i=begin;i<end;i++) {
      Element elt = self.mesh.elements[i];
      clusterBounds.extend(self.mesh.getBounds(elt));
    }

    // bool dbg = primID < 10;
    
    bounds = getBox(clusterBounds);


    vec3f point = vec3f(44.499, 108.007, 126.998);
    box3f box = getBox(clusterBounds);
    
    bool containsPoint = box.contains(point);
    bool dbg
      = containsPoint
      || (primID == 90183);
    
    
    // if (dbg) {
    //   printf("cluster %i range %i:%i is (%f %f %f):(%f %f %f)/(%f:%f), contains? %i\n",
    //          primID,
    //          begin,end,
    //          clusterBounds.lower.x,
    //          clusterBounds.lower.y,
    //          clusterBounds.lower.z,
    //          clusterBounds.upper.x,
    //          clusterBounds.upper.y,
    //          clusterBounds.upper.z,
    //          clusterBounds.lower.w,
    //          clusterBounds.upper.w,
    //          int(containsPoint));
      // printf("box (%f %f %f)(%f %f %f) point %f %f %f contains %i\n",
      //        box.lower.x,
      //        box.lower.y,
      //        box.lower.z,
      //        box.upper.x,
      //        box.upper.y,
      //        box.upper.z,
      //        point.x,
      //        point.y,
      //        point.z,
      //        int(containsPoint));
             
      // for (int i=begin;i<end;i++) {
      //   Element elt = self.mesh.elements[i];
      //   box4f eltBounds = self.mesh.getBounds(elt);
      //   printf(" > elt %i:%i (%f %f %f)(%f %f %f) sz (%f %f %f)\n",
      //          primID,i,
      //          eltBounds.lower.x,
      //          eltBounds.lower.y,
      //          eltBounds.lower.z,
      //          eltBounds.upper.x,
      //          eltBounds.upper.y,
      //          eltBounds.upper.z,
      //          eltBounds.size().x,
      //          eltBounds.size().y,
      //          eltBounds.size().z
      //          );
      // }
    // }       
    Cluster &cluster = self.clusters[primID];
    cluster.bounds = clusterBounds;
    
    if (self.xf.numValues > 0) {
      cluster.majorant = self.xf.majorant(getRange(clusterBounds),dbg);
      // if (dbg) printf("domain %f %f majorant is %f\n",self.xf.domain.lower,self.xf.domain.upper,
      //                 cluster.majorant);
      if (cluster.majorant == 0.f)
        bounds = box3f(bounds.center());
    }
    
    // if (length(bounds.span())>30) {
    // printf("bounds %f %f %f : %f %f %f\n",
    //        bounds.lower.x,
    //        bounds.lower.y,
    //        bounds.lower.z,
    //        bounds.upper.x,
    //        bounds.upper.y,
    //        bounds.upper.z);
    // }
  }

  OPTIX_CLOSEST_HIT_PROGRAM(UMeshQCCH)()
  {
    auto &ray = owl::getPRD<Ray>();
    auto &self = owl::getProgramData<UMeshQC::DD>();
    int primID = optixGetPrimitiveIndex();

    // ray.hadHit = true;
    // ray.color = .8f;//owl::randomColor(primID);
    ray.primID = primID;
    ray.tMax = optixGetRayTmax();

  }

  OPTIX_INTERSECT_PROGRAM(UMeshQCIsec)()
  {
    const int primID = optixGetPrimitiveIndex();
    const auto &self
      = owl::getProgramData<UMeshQC::DD>();
    auto &ray
      = owl::getPRD<Ray>();
    bool dbg = ray.centerPixel;
    
    Cluster cluster = self.clusters[primID];
    
    
    const vec3f org  = optixGetObjectRayOrigin();
    const vec3f dir  = optixGetObjectRayDirection();
    float t0 = optixGetRayTmin();
    float t1 = optixGetRayTmax();
    bool isHittingTheBox
      = boxTest(t0,t1,cluster.bounds,org,dir);
    if (dbg)
      printf("======== intersect primID %i range %f %f box (%.3f %.3f %.3f)(%.3f %.3f %.3f) enter (%.3f %.3f %.3f)\n",primID,
             t0,t1,
                    cluster.bounds.lower.x,
                    cluster.bounds.lower.y,
                    cluster.bounds.lower.z,
                    cluster.bounds.upper.x,
                    cluster.bounds.upper.y,
                    cluster.bounds.upper.z,
                    org.x+t0*dir.x,
                    org.y+t0*dir.y,
                    org.z+t0*dir.z
                    );
    if (!isHittingTheBox) {
      if (dbg) printf(" -> miss bounds\n");
      return;
    }

#if CLUSTERS_FROM_QC
    int begin = primID * UMeshQC::clusterSize;
    int end   = min(begin+UMeshQC::clusterSize,self.mesh.numElements);
#else
    int begin = cluster.begin;
    int end   = cluster.end;
#endif
    
    // Random rand(ray.rngSeed++,primID);
    LCG<4> &rand = (LCG<4> &)ray.rngSeed;
#if 1
    int numStepsTaken = 0, numSamplesTaken = 0, numRangesComputed = 0, numSamplesRejected = 0;
    ElementIntersector isec(self,ray,range1f(t0,t1));
    // use prim box only to find candidates (similar to ray tracing
    // for triangles), but each ray then intersects each prim
    // individually.
    int it = begin;
    Element hit_elt;
    float hit_t = INFINITY;
    while (it < end) {
      // find next prim:
      int next = it++;
      // if (dbg) printf("------ new prim\n");
      isec.setElement(self.mesh.elements[next]);
                      
       if (dbg) printf("element %i\n",isec.element.ID);
      
      // check for GEOMETRIC overlap of ray and prim
      numRangesComputed++;
      if (!isec.computeElementRange())
        continue;

      // compute majorant for given overlap range
      float majorant = isec.computeRangeMajorant();
      if (dbg) printf("element range %f %f majorant is %f\n",
                      isec.elementTRange.lower,
                      isec.elementTRange.upper,
                      majorant);
      if (majorant == 0.f)
        continue;
      
      float t = isec.elementTRange.lower;
      while (true) {
        float dt = - logf(1-rand())/(majorant);
        t += dt;
        numStepsTaken++;
        if (t >= isec.elementTRange.upper)
          break;

        isec.P = ray.org + t * ray.dir;
        numSamplesTaken++;
        isec.sampleAndMap(dbg);
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
          if (dbg)printf("REJECTED w = %f, rnd = %f, majorant %f\n",
                         isec.mapped.w,r,majorant);
          numSamplesRejected++;
          continue;
        }
        
        hit_t = t;
        hit_elt = isec.element;
        isec.leafRange.upper = hit_t;
        if (dbg) printf("**** ACCEPTED at t = %f\n",
                        t);
        break;
      }
      
    }
    if (dbg)
      printf("num ranges computed %i steps taken %i samples taken %i rejected %i\n",
             numRangesComputed, numStepsTaken, numSamplesTaken,
             numSamplesRejected);
    if (hit_t < optixGetRayTmax()) {
      // TODO : compute gradient in element only?
      MeshSampler dx0(self);
      MeshSampler dy0(self);
      MeshSampler dz0(self);
      MeshSampler dx1(self);
      MeshSampler dy1(self);
      MeshSampler dz1(self);
      const float delta = .1f;
      dx0.P = isec.P - delta * vec3f(1.f,0.f,0.f);
      dy0.P = isec.P - delta * vec3f(0.f,1.f,0.f);
      dz0.P = isec.P - delta * vec3f(0.f,0.f,1.f);
      dx1.P = isec.P + delta * vec3f(1.f,0.f,0.f);
      dy1.P = isec.P + delta * vec3f(0.f,1.f,0.f);
      dz1.P = isec.P + delta * vec3f(0.f,0.f,1.f);
      if (!dx0.sampleAndMap(begin,end))
        dx0 = isec;
      if (!dx1.sampleAndMap(begin,end))
        dx1 = isec;
      if (!dy0.sampleAndMap(begin,end))
        dy0 = isec;
      if (!dy1.sampleAndMap(begin,end))
        dy1 = isec;
      if (!dz0.sampleAndMap(begin,end))
        dz0 = isec;
      if (!dz1.sampleAndMap(begin,end))
        dz1 = isec;
      
      vec3f N;
      N.x = safeDiv(dx1.mapped.w-dx0.mapped.w,dx1.P.x - dx0.P.x);
      N.y = safeDiv(dy1.mapped.w-dy0.mapped.w,dy1.P.y - dy0.P.y);
      N.z = safeDiv(dz1.mapped.w-dz0.mapped.w,dz1.P.z - dz0.P.z);
      N = normalize
        ((N == vec3f(0.f)) ? dir : N);
      
      ray.hadHit = 1;
      ray.tMax   = hit_t;
      ray.color
        = vec3f(isec.mapped.x,isec.mapped.y,isec.mapped.z)
        * (.3f+.7f*fabsf(dot(normalize(dir),normalize(N))));
      optixReportIntersection(hit_t, 0);
    }
    
#else

#if CLUSTERS_FROM_QC
    int begin = primID * UMeshQC::clusterSize;
    int end   = min(begin+UMeshQC::clusterSize,self.mesh.numElements);
#else
    int begin = cluster.begin;
    int end   = cluster.end;
#endif
    
    MeshSampler isec(self);
    float hit_t = INFINITY;
    
    // step along the leaf box, each sample then goes against all prims equally
    int numStepsTaken = 0;
    float t = t0;
    if (dbg) printf(" -> stepping range %f %f, majorant %f\n",
                    t0,t1,cluster.majorant);
    while (true) {
      float dt = - logf(1-rand())/(cluster.majorant);
      t += dt;
      ++numStepsTaken;
      if (t >= t1)
        break;

      isec.P = org+t*dir;
      if (!isec.sampleAndMap(begin,end))
        continue;

      bool accept = (isec.mapped.w > rand()*cluster.majorant);
      if (!accept)
        continue;

      MeshSampler dx0(self);
      MeshSampler dy0(self);
      MeshSampler dz0(self);
      MeshSampler dx1(self);
      MeshSampler dy1(self);
      MeshSampler dz1(self);
      const float delta = .1f;
      dx0.P = isec.P - delta * vec3f(1.f,0.f,0.f);
      dy0.P = isec.P - delta * vec3f(0.f,1.f,0.f);
      dz0.P = isec.P - delta * vec3f(0.f,0.f,1.f);
      dx1.P = isec.P + delta * vec3f(1.f,0.f,0.f);
      dy1.P = isec.P + delta * vec3f(0.f,1.f,0.f);
      dz1.P = isec.P + delta * vec3f(0.f,0.f,1.f);
      if (!dx0.sampleAndMap(begin,end))
        dx0 = isec;
      if (!dx1.sampleAndMap(begin,end))
        dx1 = isec;
      if (!dy0.sampleAndMap(begin,end))
        dy0 = isec;
      if (!dy1.sampleAndMap(begin,end))
        dy1 = isec;
      if (!dz0.sampleAndMap(begin,end))
        dz0 = isec;
      if (!dz1.sampleAndMap(begin,end))
        dz1 = isec;
      
      vec3f N;
      N.x = safeDiv(dx1.mapped.w-dx0.mapped.w,dx1.P.x - dx0.P.x);
      N.y = safeDiv(dy1.mapped.w-dy0.mapped.w,dy1.P.y - dy0.P.y);
      N.z = safeDiv(dz1.mapped.w-dz0.mapped.w,dz1.P.z - dz0.P.z);
      N = normalize
        ((N == vec3f(0.f)) ? dir : N);
      
      ray.hadHit = 1;
      ray.tMax   = t;
      ray.color
        = vec3f(isec.mapped.x,isec.mapped.y,isec.mapped.z)
        * (.3f+.7f*fabsf(dot(normalize(dir),normalize(N))));
      optixReportIntersection(t, 0);

      if (dbg)
        printf(" accepted sample at %f, steps taken %i\n",t,numStepsTaken);
      
      return;
    }
    if (dbg)
      printf(" did not accept any sample, steps taken %i\n",numStepsTaken);
#endif
  }
#endif  
}
