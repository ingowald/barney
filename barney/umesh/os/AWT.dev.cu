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
#include "rtcore/TraceInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

enum { AWT_STACK_DEPTH = 64 };

// #define AWT_SAMPLE_THRESHOLD 4.f
// #define JOINT_FIRST_STEP 1

namespace BARNEY_NS {

  inline __rtc_device
  bool verySmall(range1f r)
  {
    float l = r.span();
    return l >= 0.f && l < 1e-5f * r.upper;
  }
  
  struct AWTPrograms {
    static inline __rtc_device
    void closestHit(rtc::TraceInterface &rt)
    {}

    static inline __rtc_device
    void anyHit(rtc::TraceInterface &rt)
    {}

    static inline __rtc_device
    void bounds(const rtc::TraceInterface &rt,
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

    static inline __rtc_device
    void intersect(rtc::TraceInterface &ti);
  };

  
  /*! approximates a cubic function defined through four points (at
    t=0, t=1/3, t=2/3, and 1=1.f) with corresponding values of f0,
    f1, f2, and f3 */
  struct Cubic {
    inline __rtc_device float eval(float t, bool dbg=false) const
    {
      t = tRange.span()==0.f
        ? 0.f
        : ((t-tRange.lower)/tRange.span());
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

  inline __rtc_device
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

  inline __rtc_device
  float evalToPlane(vec3f P, 
                    vec3f a, vec3f b, vec3f c)
  {
    vec3f N = cross(b-a,c-a);
    return dot(P-a,N);
  }
  
  inline __rtc_device
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
    
  inline __rtc_device
  Cubic cubicFromTet(const UMeshField::DD &mesh,
                     Element elt,
                     const vec3f &org,
                     const vec3f &dir,
                     range1f initRange,
                     bool dbg
                     )
  {
    Cubic cubic;
    cubic.tRange = initRange;
    
    // printf("tet ofs0 %i\n",elt.ofs0);
    vec4i tet = *(const vec4i *)&mesh.indices[elt.ofs0];
    // printf("tet ofs0 %i -> %i %i %i %i\n",elt.ofs0,
    //        tet.x,tet.y,tet.z,tet.w);
    vec4f v0 = rtc::load(((rtc::float4*)mesh.vertices)[tet.x]);
    vec4f v1 = rtc::load(((rtc::float4*)mesh.vertices)[tet.y]);
    vec4f v2 = rtc::load(((rtc::float4*)mesh.vertices)[tet.z]);
    vec4f v3 = rtc::load(((rtc::float4*)mesh.vertices)[tet.w]);
    clip(cubic.tRange,v0,v1,v2,org,dir);
    clip(cubic.tRange,v0,v3,v1,org,dir);
    clip(cubic.tRange,v0,v2,v3,org,dir);
    clip(cubic.tRange,v1,v3,v2,org,dir);
    if (dbg)
      printf("clipped range %f %f\n",
             cubic.tRange.lower,cubic.tRange.upper);
    if (cubic.tRange.lower <= cubic.tRange.upper) {
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
    inline __rtc_device
    CubicSampler(const Cubic &cubic,
                 const TransferFunction::DD &xf)
      : cubic(cubic),xf(xf)
    {}
    inline __rtc_device
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

  inline __rtc_device
  bool woodcockSampleJFS(vec4f &sample,
                         CubicSampler &sfSampler,
                         vec3f org,
                         vec3f dir,
                         range1f &tRange,
                         float majorant,
#if JOINT_FIRST_STEP
                         range1f jfsRange,
                         float jfsMajorant,
#endif
                         uint32_t &rngSeed,
                         bool dbg=false)
  {
    LCG<4> &rand = (LCG<4> &)rngSeed;
    float t = tRange.lower;

#if JOINT_FIRST_STEP
    if (t >= tRange.upper)
      return false;
      
    if (tRange.lower == jfsRange.lower)
      {
        sample = sfSampler.sampleAndMap(t,dbg);
        if (sample.w >= rand()*jfsMajorant) {
          tRange.upper = t;
          return true;
        }
      }
#endif
    
    
    while (true) {
      float dt = - logf(1.f-rand())/majorant;
      t += dt;
      
      if (t >= tRange.upper)
        return false;
      
      sample = sfSampler.sampleAndMap(t,dbg);
      if (sample.w >= rand()*majorant) {
        tRange.upper = t;
        return true;
      }
    }
  }

        
  inline __rtc_device
  void intersectPrim(const AWTAccel::DD &self,
                     vec4f &acceptedSample,
                     range1f tRange,
                     float parentMajorant,
                     vec3f org,
                     vec3f dir,
                     float &tHit,
                     int primID,
                     uint32_t &rng,
                     bool dbg=false)
  {
    Cubic cubic
      = cubicFromTet(self.mesh,self.mesh.elements[primID],org,dir,
                     tRange,
                     dbg);
    if (dbg)
      printf("tet %i %f %f\n",primID,cubic.tRange.lower,cubic.tRange.upper);
    
    if (cubic.tRange.lower > cubic.tRange.upper)
      return;
    
    if (dbg)
      printf("VALID TET\n");
    
    vec4f sample;
    CubicSampler cubicSampler(cubic,self.xf);
#ifdef AWT_SAMPLE_THRESHOLD
    if (tRange.lower == tRange.upper) {
      LCG<4> &rand = (LCG<4> &)rng;
      float t = tRange.lower;
      sample = cubicSampler.sampleAndMap(t,dbg);
      if (sample.w >= rand()*parentMajorant) {
        tHit = t;
        acceptedSample = sample;
      }
      return;
    }
#endif
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
    if (woodcockSampleJFS(sample,cubicSampler,
                          org,dir,cubic.tRange,
                          majorant,
#if JOINT_FIRST_STEP
                          jfsRange,
                          jfsMajorant,
#endif
                          rng,dbg)
        // Woodcock::sampleRangeT(sample,cubicSampler,
        //                        org,dir,cubic.tRange,
        //                        majorant,rng,dbg)
        ) {
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
  // inline __rtc_device
  // void swap(T &a, T &b)
  // {
  //   T c = a; a = b; b = c;
  // }
    
  template<bool ascending>
  inline __rtc_device
  void order(StackEntry *childEntry, int a, int b)
  {
    if (ascending  && childEntry[a].tRange.lower <= childEntry[b].tRange.lower ||
        !ascending && childEntry[a].tRange.lower >= childEntry[b].tRange.lower)
      return;
    swap(childEntry[a],childEntry[b]);
  }
  
  template<int N>
  inline __rtc_device
  void sort(StackEntry *childEntry);
  
  template<>
  inline __rtc_device
  void sort<4>(StackEntry *childEntry)
  {
    order<true>(childEntry,0,1);
    order<true>(childEntry,1,2);
    order<true>(childEntry,2,3);
    order<true>(childEntry,0,1);
    order<true>(childEntry,1,2);
    order<true>(childEntry,0,1);
  }

  inline __rtc_device
  void AWTPrograms::intersect(rtc::TraceInterface &ti)
  {
    using BARNEY_NS::render::boxTest;
    
    StackEntry stackBase[AWT_STACK_DEPTH];
    StackEntry *stack = stackBase;
    
    const void *pd = ti.getProgramData();
    
    const AWTAccel::DD &self = *(AWTAccel::DD*)pd;
    Ray &ray = *(Ray*)ti.getPRD();
#ifdef NDEBUG
    bool dbg = false;
#else
    bool dbg = ray.dbg;
#endif

    vec3f org = ti.getObjectRayOrigin();
    vec3f dir = ti.getObjectRayDirection();
    // if (!ray.dbg) return;
    
    if (dbg) {
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
      if (dbg)
        printf(" -> clip out %f %f\n",curr.tRange.lower,curr.tRange.upper);
      return;
    }

    if (dbg) {
      vec3f P = org + curr.tRange.lower*dir;
      printf("ENTERING at %f, pos %f %f %f\n",
             curr.tRange.lower,
             P.x,P.y,P.z);
    }
    curr.majorant = self.xf.baseDensity;
    curr.ref = { 0,0 };
    *stack++ = curr;

    // bool done = false;
    while (1) {
      /* repeat until we REACH A LEAF */
      while (1) {
        /* repeat until we successfully POPPED SOMETHING */
        while (1) {
          if (dbg) printf("popping at depth %i\n",
                              int(stack-stackBase));
          if (stack == stackBase) {
            if (tHit < ray.tMax) {
              ray.setVolumeHit(org+tHit*dir,
                               tHit,(const vec3f&)sample);
            }
            return;
            // done = true;
            // break;
          } 

          curr = *--stack;
          curr.tRange.upper = min(curr.tRange.upper,tHit);
          if (dbg)
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

#ifdef AWT_SAMPLE_THRESHOLD
        float tLen = curr.tRange.span();
        if (tLen > 0.f) {
          float expectedNumSteps
            = tLen// * self.xf.baseDensity
            * curr.majorant;
          if (expectedNumSteps <= AWT_SAMPLE_THRESHOLD) {
            LCG<4> &rand = (LCG<4> &)ray.rngSeed;
            float dt0 = - logf(1.f-rand())/curr.majorant;
            curr.tRange.lower += dt0;
            if (curr.tRange.lower >= curr.tRange.upper)
              continue;
            
            *stack++ = curr;
            curr.tRange.upper = curr.tRange.lower;
          }
        }
#endif

        
        AWTNode node = self.awtNodes[curr.ref.offset];
        StackEntry childEntry[AWT_NODE_WIDTH];
        for (int i=0;i<AWT_NODE_WIDTH;i++) {
          childEntry[i].ref = node.child[i].nodeRef;
          childEntry[i].majorant
            =
#ifdef AWT_SAMPLE_THRESHOLD
            (curr.tRange.lower == curr.tRange.upper)
            ? curr.majorant
            :
#endif
            node.child[i].majorant;
          childEntry[i].tRange = curr.tRange;
          
          if (node.child[i].majorant == 0.f
              ||
              !node.child[i].nodeRef.valid()
              ||
              !boxTest(org,dir,
                       childEntry[i].tRange.lower,
                       childEntry[i].tRange.upper,
                       node.child[i].bounds)) 
            childEntry[i].tRange.lower = BARNEY_INF;

          if ((curr.tRange.lower < curr.tRange.upper) &&
              verySmall(childEntry[i].tRange)) {
            // printf("box test collapsed to veeeery small value [%f %f] (%.10f -> %.10f)\n",
            //        childEntry[i].tRange.lower,childEntry[i].tRange.upper,
            //        curr.tRange.span(),childEntry[i].tRange.span());
            childEntry[i].tRange.lower = nextafter(childEntry[i].tRange.lower,-1.f);
            childEntry[i].tRange.upper = nextafter(childEntry[i].tRange.upper,BARNEY_INF);
          }
              
          if (dbg) {
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
              childEntry[i].tRange.lower < tHit) {//ray.tMax) {
            if (dbg)
              printf("pushing depth %i, node %i:%i dist %f\n",
                   int(stack-stackBase),
                   childEntry[i].ref.offset,
                   childEntry[i].ref.count,
                   childEntry[i].tRange.lower);
if (stack - stackBase >= AWT_STACK_DEPTH)
printf("STACK OVERFLOW!\n");
            *stack++ = childEntry[i];
          }
        }
      }

      /* we're at a leaf */
      // if (ray.dbg)
      //   printf("----------- leaf len %f!\n",curr.tRange.span());
      
#if JOINT_FIRST_STEP
      LCG<4> &rand = (LCG<4> &)ray.rngSeed;
      float dt0 = - logf(1.f-rand())/curr.majorant;
      curr.tRange.lower += dt0;
      if (curr.tRange.lower > curr.tRange.upper)
        continue;
#endif

      for (int i=0;i<curr.ref.count;i++) {
        curr.tRange.upper = min(curr.tRange.upper,tHit);
        intersectPrim(self,sample,
                      curr.tRange,
                      curr.majorant,
                      org,dir,tHit,self.primIDs[curr.ref.offset+i],
                      ray.rngSeed,dbg);
        curr.tRange.upper = min(curr.tRange.upper,tHit);
      }
    }
  }
  
  RTC_EXPORT_USER_GEOM(AWT,AWTAccel::DD,AWTPrograms,false,false);
} // ::BARNEY_NS

