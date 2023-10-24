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

#include "barney/unstructured/QuickClusters.h"
#include "owl/owl_device.h"

namespace barney {

  inline __device__
  vec4f DeviceXF::map(float s) const
  {
    float f = (s-domain.lower)/domain.span();
    f = clamp(f,0.f,1.f);
    f *= (numValues-1);
    int idx0 = clamp(int(f),0,numValues-1);
    int idx1 = clamp(idx0+1,0,numValues-1);
    f -= idx0;
    vec4f r = (1.f-f)*(vec4f)values[idx0]+f*(vec4f)values[idx1];
    r.w *= baseDensity;
    return r;
  }

  struct MeshSampler
  {
    inline __device__
    MeshSampler(const UMeshQC::DD &mesh) : mesh(mesh) {}

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
      
    UMeshQC::Element element;
    vec3f P;

    float scalar;
    /*! color- and alpha-mapped sample */
    vec4f mapped;
    // vec3f gradient;

    const UMeshQC::DD &mesh;    
  };

  template<typename T>
  inline __device__
  void swap(T &a, T &b) { T c = a; a = b; b = c; }

  inline __device__
  float safeDiv(float a, float b) { return (b==0.f)?0.f:(a/b); }
  
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
  bool MeshSampler::sampleAndMap() 
  {
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
      mapped = mesh.xf.map(scalar);
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
    if (dbg) printf("maj range %f %f indices %i %i\n",r.lower,r.upper,idx0,idx1);
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
    int begin = primID * UMeshQC::clusterSize;
    int end   = min(begin+UMeshQC::clusterSize,self.numElements);
    for (int i=begin;i<end;i++) {
      UMeshQC::Element elt = self.elements[i];
      clusterBounds.extend(self.getBounds(elt));
    }

    bool dbg = primID < 10;
    
    bounds = getBox(clusterBounds);

    if (dbg) printf("clusterbounds %f %f %f:%f - %f %f %f:%f, xfnum %i\n",
                    clusterBounds.lower.x,
                    clusterBounds.lower.y,
                    clusterBounds.lower.z,
                    clusterBounds.lower.w,
                    clusterBounds.upper.x,
                    clusterBounds.upper.y,
                    clusterBounds.upper.z,
                    clusterBounds.upper.w,
                    self.xf.numValues);
                    
    Cluster &cluster = self.clusters[primID];
    cluster.bounds = clusterBounds;
    
    if (self.xf.numValues > 0) {
      cluster.majorant = self.xf.majorant(getRange(clusterBounds),dbg);
      if (dbg) printf("domain %f %f majorant is %f\n",self.xf.domain.lower,self.xf.domain.upper,
                      cluster.majorant);
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
    ray.tMax = optixGetRayTmax();

  }

  OPTIX_INTERSECT_PROGRAM(UMeshQCIsec)()
  {
    const int primID = optixGetPrimitiveIndex();
    const auto &self
      = owl::getProgramData<UMeshQC::DD>();
    
    const vec3f org  = optixGetObjectRayOrigin();
    const vec3f dir  = optixGetObjectRayDirection();
    float ray_t0     = optixGetRayTmin();
    float ray_t1     = optixGetRayTmax();

    float hit_t = INFINITY;
    
    Cluster cluster = self.clusters[primID];
    float t0 = ray_t0;
    float t1 = min(ray_t1,hit_t);
    if (!boxTest(t0,t1,cluster.bounds,org,dir))
      return;

    MeshSampler isec(self);
    int begin = primID * UMeshQC::clusterSize;
    int end   = min(begin+UMeshQC::clusterSize,self.numElements);
    
    auto &ray = owl::getPRD<Ray>();
    Random rand(ray.rngSeed++,primID);

    float t = t0;
    while (true) {
      float dt = - logf(1-rand())/(cluster.majorant);
      t += dt;
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
      return;
    }
  }
  
}
