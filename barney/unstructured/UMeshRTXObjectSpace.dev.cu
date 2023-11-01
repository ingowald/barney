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

#include "barney/unstructured/UMeshRTXObjectSpace.h"
#include "owl/owl_device.h"

namespace barney {

  using Cluster = UMeshRTXObjectSpace::Cluster;

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
        scalar[i] = -INFINITY;

      eval(begin,end);
    }

    struct Plane {
      inline __device__
      Plane(float4 a, float4 b, float4 c)
      {
        p = vec4f(a);
        N = cross(getPos(vec4f(b))-getPos(a),
                  getPos(vec4f(c))-getPos(a));
      }
      
      vec3f N;
      vec4f p;
    };
    
    inline __device__
    void eval(int begin, int end)
    {
      for (int ei=begin;ei<end;ei++) {
        // if (dbg) printf("evaluating element %i [%i %i]\n",
        //                 ei,begin,end);
        Element elt = dd.elements[ei];
        if (elt.type != Element::TET)
          continue;

        if (evalTet(elt.ID))
          break;
      }
      float mapped_scalar[6];
      // if (scalar[6] < 0.f)
      //   if (dbg)
      //     printf("NO HIT IN CENTER!?!?!?!\n");
#pragma unroll
      for (int i=0;i<6;i++) {
        if (scalar[i] < 0.f)
          { scalar[i] = scalar[6]; P[i] = P[6]; }
        mapped_scalar[i] = xf.map(scalar[i]).w;
      }
#if 0
      N.x = safeDiv(scalar[1]-scalar[0], P[1].x-P[0].x);
      N.y = safeDiv(scalar[3]-scalar[2], P[3].y-P[2].y);
      N.z = safeDiv(scalar[5]-scalar[4], P[5].z-P[4].z);
#else
      N.x = safeDiv(mapped_scalar[1]-mapped_scalar[0], P[1].x-P[0].x);
      N.y = safeDiv(mapped_scalar[3]-mapped_scalar[2], P[3].y-P[2].y);
      N.z = safeDiv(mapped_scalar[5]-mapped_scalar[4], P[5].z-P[4].z);
#endif
      mappedColor = getPos(xf.map(scalar[6]));
      // N.x = safeDiv(dx1.mapped.w-dx0.mapped.w,dx1.P.x - dx0.P.x);
      // N.y = safeDiv(dy1.mapped.w-dy0.mapped.w,dy1.P.y - dy0.P.y);
      // N.z = safeDiv(dz1.mapped.w-dz0.mapped.w,dz1.P.z - dz0.P.z);
    }
    
    inline __device__
    void evalPlane(Plane plane, float v)
    {
#pragma unroll
      for (int i=0;i<7;i++) {
        float f = dot(P[i] - getPos(plane.p),plane.N);
        if (f < 0.f) {
          sw[i] = -INFINITY;
        } else {
          sw[i] += f;
          sv[i] += v * f;
        }
      }
    }
    
    inline __device__
    bool evalTet(int tetID)
    {
      // if (dbg) printf("cd: evaluating tet %i\n",tetID);
      
      int4 indices = dd.tetIndices[tetID];
      v0 = dd.vertices[indices.x];
      v1 = dd.vertices[indices.y];
      v2 = dd.vertices[indices.z];
      v3 = dd.vertices[indices.w];
      
#pragma unroll
      for (int i=0;i<7;i++)
        sw[i] = sv[i] = 0.f;
      
      evalPlane(Plane(v0,v1,v2),v3.w);
      evalPlane(Plane(v0,v3,v1),v2.w);
      evalPlane(Plane(v0,v2,v3),v1.w);
      evalPlane(Plane(v1,v3,v2),v0.w);
      
#pragma unroll
      for (int i=0;i<7;i++)
        if (sw[i] >= 0.f)
          scalar[i] = sv[i] / sw[i];
      
      bool done = true;
#pragma unroll
      for (int i=0;i<7;i++)
        if (scalar[i] < 0.f)
          done = false;
      
      return done;
    }
    
    vec3f P[7];
    float scalar[7];
    float sw[7], sv[7];
    vec3f N;
    float4 v0, v1, v2, v3;
    const bool dbg;
    vec3f mappedColor;
    const UMeshField::DD &dd;
    const TransferFunction::DD &xf;
  };
  
  struct ElementIntersector
  {
    inline __device__
    ElementIntersector(const UMeshRTXObjectSpace::DD &dd,
                       Ray &ray,
                       range1f leafRange
                       )
      : dd(dd), ray(ray), leafRange(leafRange)
    {}

    inline __device__
    void sampleAndMap(vec3f P, bool dbg);

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
    float evalTet(vec3f P, bool dbg = false)
    {
      float t3 = evalToPlane(P,v0,v1,v2);
      float t2 = evalToPlane(P,v0,v3,v1);
      float t1 = evalToPlane(P,v0,v2,v3);
      float t0 = evalToPlane(P,v1,v3,v2);

      float scale = 1.f/(t0+t1+t2+t3);
      // if (dbg) printf("eval %f %f %f %f check %f -> scale %f, vtx.w %f %f %f %f\n",
      //                 t0,t1,t2,t3,evalToPlane(getPos(v3),v0,v1,v2),scale,
      //                 v0.w,v1.w,v2.w,v3.w);
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
      // if (dbg) printf("eltrange at start %f %f\n",
      //                 elementTRange.lower,
      //                 elementTRange.upper);
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
    vec4f v0, v1, v2, v3;

    const UMeshRTXObjectSpace::DD &dd;
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
#if 0
    if (a == 0.f) return false;
    if (a < 0.f) {
      swap(v0,v1);
      swap(s0,s1);
      a = -a;
    }
#endif
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

  // inline __device__
  // bool MeshSampler::sample() 
  // {
  //   auto &mesh = dd.mesh;
  //   if (element.type == UMeshRTXObjectSpace::TET) {
  //     auto indices = mesh.tetIndices[element.ID];
  //     vec4f a = mesh.vertices[indices.x];
  //     vec4f b = mesh.vertices[indices.y];
  //     vec4f c = mesh.vertices[indices.z];
  //     vec4f d = mesh.vertices[indices.w];
  //     return evaluateTet(getPos(a),a.w,
  //                        getPos(b),b.w,
  //                        getPos(c),c.w,
  //                        getPos(d),d.w,
  //                        scalar, P);
  //   }
  //   return false;
  // }
  
  inline __device__
  void ElementIntersector::sampleAndMap(vec3f P, bool dbg) 
  {
    scalar = evalTet(P,dbg);
    mapped = dd.xf.map(scalar,dbg);
    // if (dbg)
    //   printf("***** scalar %f mapped %f %f %f: %f\n",scalar,
    //          mapped.x,mapped.y,mapped.z,mapped.w);
  }
  

// #if 0  
//   inline __device__
//   bool MeshSampler::sampleAndMap() 
//   {
//     auto &mesh = dd.mesh;
//     if (element.type == Element::TET) {
//       auto indices = mesh.tetIndices[element.ID];
//       vec4f a = mesh.vertices[indices.x];
//       vec4f b = mesh.vertices[indices.y];
//       vec4f c = mesh.vertices[indices.z];
//       vec4f d = mesh.vertices[indices.w];
//       if (!evaluateTet(getPos(a),a.w,
//                        getPos(b),b.w,
//                        getPos(c),c.w,
//                        getPos(d),d.w,
//                        scalar, P))
//         return false;
//       mapped = dd.xf.map(scalar);
//       // gradient
//       //   = (mesh.xf.map(a.w).w-mapped.w)*(getPos(a)-P)
//       //   + (mesh.xf.map(b.w).w-mapped.w)*(getPos(b)-P)
//       //   + (mesh.xf.map(c.w).w-mapped.w)*(getPos(c)-P)
//       //   + (mesh.xf.map(d.w).w-mapped.w)*(getPos(d)-P);
      
//       return true;
//     }
//     return false;
//   }
  

//   inline __device__
//   bool MeshSampler::sampleAndMap(int elts_begin, int elts_end) 
//   {
//     auto &mesh = dd.mesh;
//     for (int i=elts_begin;i<elts_end;i++) {
//       element = mesh.elements[i];
//       if (sampleAndMap())
//         return true;
//     }
//     return false;
//   }

// #endif
  
      
  
  // inline __device__
  // float DeviceXF::majorant(range1f r, bool dbg) const
  // {
  //   float f_lo = (r.lower-domain.lower)/domain.span();
  //   float f_hi = (r.upper-domain.lower)/domain.span();
  //   f_lo = clamp(f_lo,0.f,1.f);
  //   f_hi = clamp(f_hi,0.f,1.f);
  //   f_lo *= (numValues-1);
  //   f_hi *= (numValues-1);
  //   int idx0 = clamp(int(f_lo),0,numValues-1);
  //   int idx1 = clamp(int(f_hi)+1,0,numValues-1);
  //   float m = 0.f;
  //   for (int i=idx0;i<=idx1;i++)
  //     m = max(m,values[i].w);
  //   // printf("maj [%f %f] domain [%f %f]-> idx [%i %i] max %f dens %f\n",
  //   //        r.lower,r.upper,domain.lower,domain.upper,idx0,idx1,m,baseDensity);
  //   return m * baseDensity;
  // }

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
  
  OPTIX_BOUNDS_PROGRAM(UMeshRTXObjectSpaceBounds)(const void *geomData,                
                                                  owl::common::box3f &bounds,  
                                                  const int32_t primID)
  {
    const auto &self = *(const UMeshRTXObjectSpace::DD *)geomData;
    Cluster &cluster = self.clusters[primID];
    int begin = self.clusters[primID].begin;
    int end   = self.clusters[primID].end;

    if (self.xf.values == 0) {
      // first time build - all prims are active, xf and cluster
      // bounds not yet set
      cluster.bounds = box4f();
      for (int i=begin;i<end;i++)
        cluster.bounds.extend(self.mesh.eltBounds(self.mesh.elements[i]));
    }

    bounds = getBox(cluster.bounds);

    if (self.xf.values) {
      cluster.majorant = self.xf.majorant(getRange(cluster.bounds));
      if (cluster.majorant == 0.f)
        bounds = box3f(bounds.center());
    }
  }

  OPTIX_CLOSEST_HIT_PROGRAM(UMeshRTXObjectSpaceCH)()
  {
    auto &ray = owl::getPRD<Ray>();
    auto &self = owl::getProgramData<UMeshRTXObjectSpace::DD>();
    int primID = optixGetPrimitiveIndex();
    Cluster &cluster = self.clusters[primID];
    
    // ray.hadHit = true;
    // ray.color = .8f;//owl::randomColor(primID);
    // ray.primID = primID;
    ray.tMax = optixGetRayTmax();

    vec3f P = ray.org + ray.tMax * ray.dir;
    // if (ray.dbg)
    //   printf("CENTRAL DIFF prim %i at %f %f %f\n",
    //          primID,
    //          P.x,P.y,P.z);
    CentralDifference cd(self.mesh,self.xf,P,cluster.begin,cluster.end,ray.dbg);
    
    // eval(cd,self.mesh,self.clusters[primID].begin,self.clusters[primID].end);

    vec3f N = normalize
      ((cd.N == vec3f(0.f)) ? ray.dir : cd.N);

    // if (ray.dbg)
    //   printf("cd.N %f %f %f, dot %f\n",
    //          cd.N.x,
    //          cd.N.y,
    //          cd.N.z,
    //          fabsf(dot(normalize(ray.dir),normalize(N))));

    ray.hadHit = 1;
    ray.color
      = //randomColor(primID)
      cd.mappedColor
      * (.3f+.7f*fabsf(dot(normalize(ray.dir),normalize(N))));
  }

  OPTIX_INTERSECT_PROGRAM(UMeshRTXObjectSpaceIsec)()
  {
    const int primID = optixGetPrimitiveIndex();
    const auto &self
      = owl::getProgramData<typename UMeshRTXObjectSpace::DD>();
    auto &ray
      = owl::getPRD<Ray>();
    bool dbg = 0;//ray.dbg;
    
    Cluster cluster = self.clusters[primID];
    
    
    const vec3f org  = optixGetObjectRayOrigin();
    const vec3f dir  = optixGetObjectRayDirection();
    float t0 = optixGetRayTmin();
    float t1 = optixGetRayTmax();
    bool isHittingTheBox
      = boxTest(t0,t1,cluster.bounds,org,dir);
    // if (dbg)
    //   printf("======== intersect primID %i range %f %f box (%.3f %.3f %.3f)(%.3f %.3f %.3f) enter (%.3f %.3f %.3f)\n",primID,
    //          t0,t1,
    //          cluster.bounds.lower.x,
    //          cluster.bounds.lower.y,
    //          cluster.bounds.lower.z,
    //          cluster.bounds.upper.x,
    //          cluster.bounds.upper.y,
    //          cluster.bounds.upper.z,
    //          org.x+t0*dir.x,
    //          org.y+t0*dir.y,
    //          org.z+t0*dir.z
    //          );
    if (!isHittingTheBox) {
      // if (dbg) printf(" -> miss bounds\n");
      return;
    }

#if CLUSTERS_FROM_QC
    int begin = primID * UMeshRTXObjectSpace::clusterSize;
    int end   = min(begin+UMeshRTXObjectSpace::clusterSize,self.mesh.numElements);
#else
    int begin = cluster.begin;
    int end   = cluster.end;
#endif
    
    // Random rand(ray.rngSeed++,primID);
    LCG<4> &rand = (LCG<4> &)ray.rngSeed;
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
                      
      // if (dbg) printf("element %i\n",isec.element.ID);
      
      // check for GEOMETRIC overlap of ray and prim
      numRangesComputed++;
      if (!isec.computeElementRange())
        continue;

      // compute majorant for given overlap range
      float majorant = isec.computeRangeMajorant();
      // if (dbg) printf("element range %f %f majorant is %f\n",
      //                 isec.elementTRange.lower,
      //                 isec.elementTRange.upper,
      //                 majorant);
      if (majorant == 0.f)
        continue;
      
      float t = isec.elementTRange.lower;
      while (true) {
        float dt = - logf(1-rand())/(majorant);
        t += dt;
        numStepsTaken++;
        if (t >= isec.elementTRange.upper)
          break;

        vec3f P = ray.org + t * ray.dir;
        numSamplesTaken++;
        isec.sampleAndMap(P,dbg);
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
        // if (dbg) printf("**** ACCEPTED at t = %f, P = %f %f %f, tet ID %i\n",
        //                 t, P.x,P.y,P.z,isec.element.ID);
        break;
      }
      
    }
    // if (dbg)
    //   printf("num ranges computed %i steps taken %i samples taken %i rejected %i\n",
    //          numRangesComputed, numStepsTaken, numSamplesTaken,
    //          numSamplesRejected);


    if (hit_t < optixGetRayTmax())  {
      // if (ray.dbg)
      //   printf("ISEC at prim %i, tet %i, t %f\n",
      //          primID,isec.element.ID,hit_t);
      optixReportIntersection(hit_t, 0);
    }
  }
}
