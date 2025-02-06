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

#include <owl/owl_device.h>
#include "barney/umesh/os/AWT.h"

RTC_DECLARE_GLOBALS(barney::render::OptixGlobals);

enum { AWT_STACK_DEPTH = 256 };

namespace barney {

  struct AWTPrograms {
    template<typename TraceInterface>
    static inline __both__
    void closest_hit(TraceInterface &rt)
    {}

    template<typename TraceInterface>
    static inline __both__
    void any_hit(TraceInterface &rt)
    {}

    template<typename RTBackend>
    static inline __both__
    void bounds(const RTBackend &rt,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    { 
      const AWTAccel::DD &geom = *(const AWTAccel::DD *)geomData;
      bounds = geom.mesh.worldBounds;

      printf("BOUNDS (%f %f %f)(%f %f %f)\n",
             bounds.lower.x,
             bounds.lower.y,
             bounds.lower.z,
             bounds.upper.x,
             bounds.upper.y,
             bounds.upper.z);
    
      
    }

    template<typename TI>
    static inline __both__
    void intersect(TI &ti);
  };

  
  /*! approximates a cubic function defined through four points (at
    t=0, t=1/3, t=2/3, and 1=1.f) with corresponding values of f0,
    f1, f2, and f3 */
  struct Cubic {
    inline __both__ float eval(float t, bool dbg=false) const
    {
      t = (t-tRange.lower)/tRange.span();
      t = 3.f*t;
      float _f0, _f1;
      if (t >= 2.f) {
        _f0 = f2;
        _f1 = f3;
        t -= 2.f;
      } else if (t >= 1.f) {
        _f0 = f1;
        _f1 = f2;
        t -= 1.f;
      } else {
        _f0 = f0;
        _f1 = f1;
      }
      return (1.f-t)*_f0 + t*_f1;
    }
    range1f tRange;
    float f0, f1, f2, f3;
  };

  inline __both__
  void clip(range1f &range,
            vec4f _a, vec4f _b, vec4f _c,
            const vec3f &org,
            const vec3f &dir)
  {
    vec3f a = getPos(_a);
    vec3f b = getPos(_b);
    vec3f c = getPos(_c);
    vec3f N = cross(b-a,c-a);
    float NdotD = dot(N,dir);
    if (NdotD == 0.f) {
      if (dot(org - a,N) < 0.f)
        range = { +BARNEY_INF, -BARNEY_INF };
      return;
    }
    float plane_t = - dot(org - a, N) / NdotD;
    if (NdotD < 0.f)
      range.upper = min(range.upper,plane_t);
    else
      range.lower = max(range.lower,plane_t);
  }

  inline __both__
  float evalToPlane(vec3f P, 
                    vec3f a, vec3f b, vec3f c)
  {
    vec3f N = cross(b-a,c-a);
    return dot(P-a,N);
  }
  
  inline __both__
  float eval(vec3f P, 
             vec4f _a, vec4f _b, vec4f _c, vec4f _d)
  {
    vec3f v0 = getPos(_a);
    vec3f v1 = getPos(_b);
    vec3f v2 = getPos(_c);
    vec3f v3 = getPos(_d);
      
    float f3 = evalToPlane(P,v0,v1,v2);
    float f2 = evalToPlane(P,v0,v3,v1);
    float f1 = evalToPlane(P,v0,v2,v3);
    float f0 = evalToPlane(P,v1,v3,v2);
    return
      (f0*_a.w + f1*_b.w + f2*_c.w + f3*_d.w)
      / (f0+f1+f2+f3);
  }
    
  inline __both__
  Cubic cubicFromTet(const UMeshField::DD &mesh,
                     Element elt,
                     const vec3f &org,
                     const vec3f &dir,
                     float &tHit,
                     bool dbg
                     )
  {
    Cubic cubic;
    cubic.tRange = {1e-6f,tHit};

    // printf("tet ofs0 %i\n",elt.ofs0);
    vec4i tet = *(const vec4i *)&mesh.indices[elt.ofs0];
    // printf("tet ofs0 %i -> %i %i %i %i\n",elt.ofs0,
    //        tet.x,tet.y,tet.z,tet.w);
    vec4f v0 = load(((float4*)mesh.vertices)[tet.x]);
    vec4f v1 = load(((float4*)mesh.vertices)[tet.y]);
    vec4f v2 = load(((float4*)mesh.vertices)[tet.z]);
    vec4f v3 = load(((float4*)mesh.vertices)[tet.w]);
    clip(cubic.tRange,v0,v1,v2,org,dir);
    clip(cubic.tRange,v0,v3,v1,org,dir);
    clip(cubic.tRange,v0,v2,v3,org,dir);
    clip(cubic.tRange,v1,v3,v2,org,dir);
    if (dbg)
      printf("clipped range %f %f\n",
             cubic.tRange.lower,cubic.tRange.upper);
    if (cubic.tRange.lower < cubic.tRange.upper) {
      vec3f P0 = org+cubic.tRange.lower*dir;
      vec3f P1 = org+cubic.tRange.upper*dir;
      cubic.f0 = eval(P0,v0,v1,v2,v3);
      cubic.f3 = eval(P1,v0,v1,v2,v3);
      cubic.f1 = lerp_l(1.f/3.f,cubic.f0,cubic.f3);
      cubic.f2 = lerp_l(2.f/3.f,cubic.f0,cubic.f3);
    }
    if (dbg)
      printf("got cubic corners %f %f %f %f f's %f %f %f %f\n",
             v0.w,
             v1.w,
             v2.w,
             v3.w,
             cubic.f0,
             cubic.f1,
             cubic.f2,
             cubic.f3);
    return cubic;
  }

  struct CubicSampler {
    inline __both__
    CubicSampler(const Cubic &cubic,
                 const TransferFunction::DD &xf)
      : cubic(cubic),xf(xf)
    {}
    inline __both__
    vec4f sampleAndMap(float t, bool dbg=false) const
    {
      float f = cubic.eval(t,dbg);
      if (dbg)
        printf("cubic sampling at t=%f in %f %f -> scalar %f\n",t,
               cubic.tRange.lower,cubic.tRange.upper,f);
      if (isnan(f)) return vec4f(0.f);
      vec4f mapped = xf.map(f,dbg);
      if (dbg)
        printf("mapping %f -> %f %f %f:%f\n",f,
               mapped.x,mapped.y,mapped.z,mapped.w);
      return mapped;
    }
    
    const Cubic &cubic;
    const TransferFunction::DD &xf;
  };
    
  inline __both__
  void intersectPrim(const AWTAccel::DD &self,
                     vec4f &acceptedSample,
                     vec3f org,
                     vec3f dir,
                     float &tHit,
                     int primID,
                     uint32_t &rng,
                     bool dbg=false)
  {
    Cubic cubic
      = cubicFromTet(self.mesh,self.mesh.elements[primID],org,dir,tHit,
                     dbg);
    if (dbg)
      printf("tet %i %f %f\n",primID,cubic.tRange.lower,cubic.tRange.upper);

    if (cubic.tRange.lower >= cubic.tRange.upper)
      return;

    if (dbg)
      printf("VALID TET\n");

    vec4f sample;
    CubicSampler cubicSampler(cubic,self.xf);
    range1f tetRange = {
      min(cubic.f0,cubic.f3),
      max(cubic.f0,cubic.f3)
    };
    float majorant = self.xf.majorant(tetRange);
    // if (majorant == 0.f) return;

    if (dbg)
      printf("->woodcock range %f %f majorant %f\n",
             cubic.tRange.lower,
           cubic.tRange.upper,
             majorant);
    if (Woodcock::sampleRangeT(sample,cubicSampler,
                               org,dir,cubic.tRange,
                               majorant,rng,dbg)) {
      tHit = cubic.tRange.upper;
      acceptedSample = sample;
      if (dbg)
        printf("ACCEPTED at %f, sample %f %f %f\n",
               tHit,sample.x,sample.y,sample.z);
    }
    
  }
  
  


  
  struct __barney_align(16) StackEntry {
    AWTNode::NodeRef ref; // 1 dword
    float   majorant;     // 1 dword
    range1f tRange;       // 2 dwords
  };

  // template<typename T>
  // inline __both__
  // void swap(T &a, T &b)
  // {
  //   T c = a; a = b; b = c;
  // }
    
  template<bool ascending>
  inline __both__
  void order(StackEntry *childEntry, int a, int b)
  {
    if (ascending  && childEntry[a].tRange.lower <= childEntry[b].tRange.lower ||
        !ascending && childEntry[a].tRange.lower >= childEntry[b].tRange.lower)
      return;
    swap(childEntry[a],childEntry[b]);
  }
  
  template<int N>
  inline __both__
  void sort(StackEntry *childEntry);
  
  template<>
  inline __both__
  void sort<4>(StackEntry *childEntry)
  {
    order<true>(childEntry,0,1);
    order<true>(childEntry,1,2);
    order<true>(childEntry,2,3);
    order<true>(childEntry,0,1);
    order<true>(childEntry,1,2);
    order<true>(childEntry,0,1);
  }

  template<typename TI>
  inline __both__
  void AWTPrograms::intersect(TI &ti)
  {
    using barney::render::boxTest;
    
    StackEntry stackBase[AWT_STACK_DEPTH];
    StackEntry *stack = stackBase;
    
    const void *pd = ti.getProgramData();
    
    const AWTAccel::DD &self = *(AWTAccel::DD*)pd;
    Ray &ray = *(Ray*)ti.getPRD();

    vec3f org = ti.getObjectRayOrigin();
    vec3f dir = ti.getObjectRayDirection();
    // if (!ray.dbg) return;
    
    if (ray.dbg) {
      printf("=========== ray at awt ===========\n");
      printf("org %f %f %f\n",org.x,org.y,org.z);
      printf("dir %f %f %f\n",dir.x,dir.y,dir.z);
      printf("self domain %f %f\n",
             self.xf.domain.lower,self.xf.domain.upper);
    }
    // else
    //   return;
    
    StackEntry curr;
    vec4f sample;
    float tHit = ti.getRayTmax();
    curr.tRange = { 1e-6f, tHit };
    
    if (!boxTest(org,dir,
                                 curr.tRange.lower,curr.tRange.upper,
                 self.mesh.worldBounds)) {
      // doesn't even overlap the bounding box... 
      if (ray.dbg)
        printf(" -> clip out %f %f\n",curr.tRange.lower,curr.tRange.upper);
      return;
    }

    if (ray.dbg) {
      vec3f P = org + curr.tRange.lower*dir;
      printf("ENTERING at %f, pos %f %f %f\n",
             curr.tRange.lower,
             P.x,P.y,P.z);
    }
    curr.majorant = self.xf.baseDensity;
    curr.ref = { 0,0 };
    *stack++ = curr;

    bool done = false;
    while (!done) {
      /* repeat until we REACH A LEAF */
      while (!done) {
        /* repeat until we successfully POPPED SOMETHING */
        while (!done) {
          if (ray.dbg) printf("popping at depth %i\n",
                              int(stack-stackBase));
          if (stack == stackBase) {
            done = true;
            break;
          }

          curr = *--stack;
          curr.tRange.upper = min(curr.tRange.upper,tHit);
          if (ray.dbg)
            printf("@ %i:%i, range %f %f maj %f\n",
                   curr.ref.offset,
                   curr.ref.count,
                   curr.tRange.lower,
                   curr.tRange.upper,
                   curr.majorant);
          if (curr.tRange.lower >= tHit)
            continue;
            
          break;
        }
        // we did POP SOMETHING
        if (curr.ref.count)
          break;

        AWTNode node = self.awtNodes[curr.ref.offset];
        StackEntry childEntry[AWT_NODE_WIDTH];
        for (int i=0;i<AWT_NODE_WIDTH;i++) {
          childEntry[i].ref = node.child[i].nodeRef;
          childEntry[i].majorant = node.child[i].majorant;
          childEntry[i].tRange = curr.tRange;
          if (!node.child[i].nodeRef.valid()
              ||
              !boxTest(org,dir,
                       childEntry[i].tRange.lower,
                       childEntry[i].tRange.upper,
                       node.child[i].bounds)) 
            childEntry[i].tRange.lower = BARNEY_INF;
          if (ray.dbg) {
            // printf(" box %i (%f %f %f)(%f %f %f)\n",
            //        i,
            //        node.child[i].bounds.lower.x,
            //        node.child[i].bounds.lower.y,
            //        node.child[i].bounds.lower.z,
            //        node.child[i].bounds.upper.x,
            //        node.child[i].bounds.upper.y,
            //        node.child[i].bounds.upper.z);
            printf(" child %2i box (%5.1f %5.1f %5.1f)(%5.1f %5.1f %5.1f) range %5.1f %5.1f\n",
                   i,
                   node.child[i].bounds.lower.x,
                   node.child[i].bounds.lower.y,
                   node.child[i].bounds.lower.z,
                   node.child[i].bounds.upper.x,
                   node.child[i].bounds.upper.y,
                   node.child[i].bounds.upper.z,
                   childEntry[i].tRange.lower,
                   childEntry[i].tRange.upper);
          }
        }
        sort<AWT_NODE_WIDTH>(childEntry);
        for (int i=AWT_NODE_WIDTH-1;i>=0;--i) {
          if (childEntry[i].majorant > 0.f
              &&
              childEntry[i].tRange.lower < ray.tMax) {
            if (ray.dbg)
              printf("pushing depth %i, node %i:%i dist %f\n",
                   int(stack-stackBase),
                   childEntry[i].ref.offset,
                   childEntry[i].ref.count,
                   childEntry[i].tRange.lower);
            *stack++ = childEntry[i];
          }
        }
      }

      /* we're at a leaf */
      if (ray.dbg)
        printf("----------- leaf!\n");
      for (int i=0;i<curr.ref.count;i++)
        intersectPrim(self,sample,
                      org,dir,tHit,self.primIDs[curr.ref.offset+i],
                      ray.rngSeed,ray.dbg);
    }
    if (tHit < ray.tMax) {
      ray.setVolumeHit(org+tHit*dir,
                       tHit,(const vec3f&)sample);
    }
  }
  
} // ::barney

RTC_DECLARE_USER_GEOM(AWT,barney::AWTPrograms);
