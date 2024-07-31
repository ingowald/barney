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

// #define AWT_SAMPLE_MODE 1

#ifndef AWT_THRESHOLD
#  define AWT_THRESHOLD 8
#endif

// #if AWT_SAMPLE_MODE
// # ifndef AWT_THRESHOLD
// #  define AWT_THRESHOLD 
// # endif
// #endif

namespace barney {
  namespace device {
  OPTIX_BOUNDS_PROGRAM(UMeshAWTBounds)(const void *geomData,                
                                       owl::common::box3f &primBounds,  
                                       const int32_t primID)
  {
    const auto &self = *(const UMeshAWT::DD *)geomData;
    box4f bounds;
    int root = self.roots[primID];
    int rootChild = root & 0x3;
    int rootNode  = root >> 2;
    // int begin = self.nodes[rootNode].child[rootChild].offset;
    bounds = self.nodes[rootNode].bounds[rootChild];
    // if (self.xf.values == 0) {
    //   for (int i=begin;i<end;i++)
    //     bounds.extend(self.eltBounds(self.elements[i]));
    //   self.nodes[rootNode].bounds[rootChild] = bounds;
    // }
    // else 
    //   bounds = self.nodes[rootNode].bounds[rootChild];
    primBounds = getBox(bounds);
    // printf("bounds prog %i root %i:%i\n",primID,rootNode,rootChild);
    range1f range = getRange(bounds);
    if (self.xf.values) {
      float majorant = self.xf.majorant(range);
      self.nodes[rootNode].majorant[rootChild] = majorant;
      if (majorant == 0.f) {
        // swap(primBounds.lower,primBounds.upper);
        primBounds = box3f();
        // printf("leaf CULLED\n");
      }
      
      
      // else
      //   printf("active node (%f %f %f)(%f %f %f)\n",
      //          primBounds.lower.x,
      //          primBounds.lower.y,
      //          primBounds.lower.z,
      //          primBounds.upper.x,
      //          primBounds.upper.y,
      //          primBounds.upper.z);
    }
  }
  

  OPTIX_CLOSEST_HIT_PROGRAM(UMeshAWTCH)()
  {
    auto &ray = owl::getPRD<Ray>();
    auto &self = owl::getProgramData<UMeshAWT::DD>();
    int primID = optixGetPrimitiveIndex();
    
    float t = optixGetRayTmax();

    vec3f P = ray.org + t * ray.dir;
    // if (ray.dbg)
    //   printf("CENTRAL DIFF prim %i at %f %f %f\n",
    //          primID,
    //          P.x,P.y,P.z);

    // vec3f N = normalize(ray.dir);
    ray.setVolumeHit(P,t,vec3f(1.f));
    // ray.hadHit = 1;
    // ray.hit.N = vec3f(0.f);
    // ray.hit.P = P;
  }

// #if AWT_SAMPLE_MODE
//   struct AWTSamples {
//     enum { max_samples = 4*AWT_THRESHOLD };
//     float t[max_samples];
//     int count = 0;
//   };


//   inline __device__
//   float intersectLeaf(Ray &ray,
//                       AWTSamples &samples,
//                       bool inSampleMode,
//                       float majorant,
//                       range1f &inputLeafRange,
//                       const UMeshObjectSpace::DD &self,
//                       int begin,
//                       int end)
//   {
//     bool dbg = ray.dbg;
//     LCG<4> &rand = (LCG<4> &)ray.rngSeed;
//     int numStepsTaken = 0, numSamplesTaken = 0, numRangesComputed = 0, numSamplesRejected = 0;
//     ElementIntersector isec(self,ray,inputLeafRange);
//     // use prim box only to find candidates (similar to ray tracing
//     // for triangles), but each ray then intersects each prim
//     // individually.
    
//     int it = begin;
//     Element hit_elt;
//     float hit_t = INFINITY;
//     while (it < end) {
//       // find next prim:
//       int next = it++;
//       if (!isec.setElement(self.elements[next]))
//         continue;

//       if (inSampleMode) {
//         for (int i=0;i<samples.count;i++) {
//           float t = samples.t[i];
//           if (t < isec.leafRange.lower || t >= min(hit_t,isec.leafRange.upper))
//             continue;
//           vec3f P = ray.org + t * ray.dir;
//           isec.sampleAndMap(P,dbg);
//           float r = rand();
//           bool accept = (isec.mapped.w > r*majorant);
//           if (!accept) {
//             continue;
//           }
//           hit_t = t;
//           hit_elt = isec.element;
//           isec.leafRange.upper = hit_t;
          
//           ray.hit.baseColor = getPos(isec.mapped);
//           break;
//         }
//         continue;
//       } 
      
//       // check for GEOMETRIC overlap of ray and prim
//       numRangesComputed++;
//       if (!isec.computeElementRange())
//         continue;

//       // compute majorant for given overlap range
//       float majorant = isec.computeRangeMajorant();
//       if (majorant == 0.f)
//         continue;
      
//       float t = isec.elementTRange.lower;
//       while (true) {
//         float dt = - logf(1-rand())/(majorant);
//         t += dt;
//         numStepsTaken++;
//         if (t >= isec.elementTRange.upper)
//           break;

//         vec3f P = ray.org + t * ray.dir;
//         numSamplesTaken++;
//         isec.sampleAndMap(P,dbg);
//         float r = rand();
//         bool accept = (isec.mapped.w > r*majorant);
//         if (!accept) {
//           numSamplesRejected++;
//           continue;
//         }
        
//         hit_t = t;
//         hit_elt = isec.element;
//         isec.leafRange.upper = hit_t;

//         ray.hit.baseColor = getPos(isec.mapped);
//         break;
//       }
//     }
//     return hit_t;
//   }
    
// #endif


  struct __barney_align(16) AWTSegment {
    AWTNode::NodeRef node;
// #if AWT_SAMPLE_MODE
//     half    majorant;
//     bool    inSampleMode;
// #else
    float   majorant;
// #endif
    range1f range;
  };

// #if AWT_SAMPLE_MODE
//   inline __device__
//   void checkSwitchingToSampleMode(LCG<4> &rand,
//                                   AWTSegment &segment,
//                                   AWTSamples &samples,
//                                   AWTSegment *&stackPtr)
//   {
//     if (segment.inSampleMode)
//       // ALREADY in sample mode...
//       return;
//     float expected_dt
//       = - logf(.5f)/((float)segment.majorant);
//     float expectedNumSteps
//       = (segment.range.upper-segment.range.lower) / expected_dt;
//     if (expectedNumSteps > (float)AWT_THRESHOLD)
//       return;

//     // oh-kay .... let's do it! let's switch to sample mode!
//     segment.inSampleMode = true;
//     samples.count = 0;
//     float t = segment.range.lower;
//     while (true) {
//       float dt = - logf(1-rand())/((float)segment.majorant);
//       t += dt;
//       if (t > segment.range.upper)
//         break;
//       samples.t[samples.count++] = t;
//       if (samples.count == samples.max_samples) {
//         *stackPtr++ = { segment.node,segment.majorant,true,
//                         { t,segment.range.upper } };
//         break;
//       }
//     }
//     if (samples.count == 0)
//       return;
//     segment.range.lower = samples.t[0];
//     segment.range.upper = samples.t[samples.count-1];
//   }
// #else
  inline __device__
  bool switchToSampleMode(AWTSegment &segment)
  {
    float expected_dt
      = - logf(.5f)/((float)segment.majorant);
    float expectedNumSteps
      = (segment.range.upper-segment.range.lower) / expected_dt;
    if (expectedNumSteps > (float)AWT_THRESHOLD)
      return false;
    return true;
  }
// #endif

  inline __device__ void orderSegments(AWTSegment &a,
                                       AWTSegment &b)
  {
    if (a.range.lower < b.range.lower) {
      AWTSegment c = a;
      a = b;
      b = c;
    }
  }
  inline __device__
  bool shortEnoughForSampling(AWTSegment segment)
  { return false; }

// #if AWT_SAMPLE_MODE
//   inline __device__
//   range1f clampToSamples(range1f range, const AWTSamples &samples)
//   {
//     range1f clamped = { INFINITY,-INFINITY };
//     for (int i=0;i<samples.count;i++) {
//       float t = samples.t[i];
//       if (t >= range.lower && t <= range.upper)
//         clamped.extend(t);
//     }
//     if (clamped.upper < clamped.lower)
//       clamped = { INFINITY, INFINITY };
//     return clamped;
//   }
// #endif

  inline __device__ bool inside(const box4f box,
                                const vec3f P)
  {
    return !(P.x < box.lower.x |
             P.y < box.lower.y |
             P.z < box.lower.z |
             P.x > box.upper.x |
             P.y > box.upper.y |
             P.z > box.upper.z);
  }
  
  inline __device__
  float findSample(const UMeshField::DD *mesh,
                   const AWTNode *nodes,
                   AWTNode::NodeRef nodeRef,
                   vec3f P,
                   bool dbg=false)
  {
    AWTNode::NodeRef stackBase[32];
    AWTNode::NodeRef *stackPtr = stackBase;
    // if (dbg) printf("findsample %f %f %f\n",P.x,P.y,P.z);
    while (true) {
      // if (dbg) printf("noderef %i %i\n",nodeRef.offset,nodeRef.count);
      while (nodeRef.count == 0) {
        auto node = nodes[nodeRef.offset];
        AWTNode::NodeRef next; next.offset = 0;
#pragma unroll
        for (int c=0;c<4;c++) {
          if (node.majorant[c] == 0.f)
            continue;
          if (!inside(node.bounds[c],P))
            continue;
          if (next.offset == 0)
            next = node.child[c];
          else {
            // if (dbg) printf("pushing to %i\n",int(stackPtr-stackBase));
            *stackPtr++ = node.child[c];
          }
        }
        if (next.offset == 0)
          { nodeRef.count = 0; break; }
        else
          { nodeRef = next; continue; }
      }

      // do leaf
      for (int i=0;i<nodeRef.count;i++) {
        auto elt = mesh->elements[nodeRef.offset+i];
        // if (dbg) printf("sampling element %i tet %i\n",nodeRef.offset+i,elt.ID);
        float retVal = NAN;
        if (mesh->eltScalar(retVal,elt,P)) {
          // if (dbg) printf("sample found %f\n",retVal);
          return retVal;
        }
      }

      if (stackPtr == stackBase) {
        // if (dbg) printf("no sample found\n");
        return NAN;
      }

      // if (dbg) printf("popping from %i\n",int(stackPtr-stackBase));
      nodeRef = *--stackPtr;
    }
  }

  OPTIX_INTERSECT_PROGRAM(UMeshAWTIsec)()
  {
    
    const int primID = optixGetPrimitiveIndex();
    const auto &self
      = owl::getProgramData<typename UMeshAWT::DD>();
    auto &ray
      = owl::getPRD<Ray>();
    // if (ray.dbg == false) return;

    // bool dbg = false;//ray.dbg;

// #if PRINT_BALLOT
//     int numActive = __popc(__ballot(1));
//     if (ray.dbg)
//       printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ isec #%i on geom %lx, leaf %i, numActive = %i\n",
//              ray.numIsecsThisRay++,
//              (void *)&self,primID,numActive);
// #endif
    
    
     // if (!ray.dbg) return;
    AWTSegment segment;
    int root = self.roots[primID];
    int rootChild = root & 0x3;
    int rootNode  = root >> 2;
    // int begin = self.nodes[rootNode].child[rootChild].offset;
    // int end = begin + self.nodes[rootNode].child[rootChild].count;
    box3f bounds = getBox(self.nodes[rootNode].bounds[rootChild]);
    // float majorant = self.nodes[rootNode].majorant[rootChild];

    segment.range.lower = optixGetRayTmin();
    segment.range.upper = optixGetRayTmax();
    segment.node = self.nodes[rootNode].child[rootChild];
    segment.majorant = self.nodes[rootNode].majorant[rootChild];
// #if AWT_SAMPLE_MODE
//     segment.inSampleMode = false;
// #endif
    float hit_t = segment.range.upper;
    vec3f org = ray.org;
    vec3f dir = ray.dir;
    bool isHittingTheBox
      = boxTest(segment.range.lower,segment.range.upper,bounds,org,dir);

    if (!isHittingTheBox) 
      return;

// #if AWT_SAMPLE_MODE
//     AWTSamples samples;
// #endif    
    LCG<4> &rand = (LCG<4> &)ray.rngSeed;
    AWTSegment stackBase[32];
    AWTSegment *stackPtr = stackBase;
    bool haveValidSegment = true;
    // if (dbg)
    //   printf("====================================\n");
    while (true) {
      bool inSamplingMode = false;
      while (true) {
        if (!haveValidSegment) {
          while (stackPtr > stackBase) {
            segment = *--stackPtr;
            if (segment.range.lower >= hit_t) 
              continue;
            // we FOUND something to pop!
            segment.range.upper = min(segment.range.upper,hit_t);
            haveValidSegment = true;
            break;
          }
        }
        // we'd STILL need to pop from stack, but stack is empty .... we're done.
        if (!haveValidSegment) 
          break;

        if (switchToSampleMode(segment)) {
          inSamplingMode = true;
          break;
        }
        
        if (segment.node.count != 0)
          /* leaf! (even though not in samplign mode...) */
          break;
        
// #if AWT_SAMPLE_MODE
        // checkSwitchingToSampleMode(rand,segment,samples,stackPtr);
        // #else
// #endif
        
        auto node = self.nodes[segment.node.offset];
        
        AWTSegment childSeg[4];
#pragma unroll(4)
        for (int c=0;c<4;c++) {
          float tt0 = segment.range.lower;
          float tt1 = segment.range.upper;
          childSeg[c]
            = { node.child[c], 
// #if AWT_SAMPLE_MODE
//             segment.inSampleMode
//             ?(float)segment.majorant
//             :node.majorant[c],
//             segment.inSampleMode,
// #else
            node.majorant[c],
// #endif
            range1f{ INFINITY, INFINITY } };
          if (node.majorant[c] == 0.f) {
          } else if (!boxTest(tt0,tt1,node.bounds[c],org,dir)) {
            // do nothing -- we didn't hit, just leave the range.lwoer at inf
          } else {
            // we DID hit the box
            childSeg[c].range = range1f{ tt0, tt1 };
// #if AWT_SAMPLE_MODE
//             // ... but may still miss any samples
//             if (segment.inSampleMode) {
//               childSeg[c].range
//                 = clampToSamples(childSeg[c].range,samples);
//             } 
// #endif
          }
        }

        orderSegments(childSeg[0],childSeg[1]);
        orderSegments(childSeg[1],childSeg[2]);
        orderSegments(childSeg[2],childSeg[3]);

        orderSegments(childSeg[0],childSeg[1]);
        orderSegments(childSeg[1],childSeg[2]);

        orderSegments(childSeg[0],childSeg[1]);

        if (childSeg[0].range.lower < hit_t) *stackPtr++ = childSeg[0];
        if (childSeg[1].range.lower < hit_t) *stackPtr++ = childSeg[1];
        if (childSeg[2].range.lower < hit_t) *stackPtr++ = childSeg[2];
        
        if (childSeg[3].range.lower >= hit_t) {
          haveValidSegment = false;
        } else {
          segment = childSeg[3];
        }
      }

      // if we reached here we either couldn't go any further down the
      // tree, or decided not to. in total there's three optoins how
      // this could have come about:
      //
      // a) traversal reached a node where none of the children are
      // valid; we can't go down and ned to pop.
      //
      // b) we've reached a valid leaf, and need to do (object-space)
      // leaf interseciton with the primitmives there.
      //
      // c) we've reached an inner node that _may_ have children, but
      // where we decided that it's small enough to just sample the
      // entire range
      //
      
      if (!haveValidSegment) 
        // option 'a' - we know the ray doesn't ahve a valid segment,
        // so nothing to do here.
        break;

      // if (dbg)
      //   printf("----------- reached segment %f %f node %i %i, samplemode = %i\n",
      //          segment.range.lower,segment.range.upper,
      //          segment.node.offset,
      //          segment.node.count,
      //          int(inSamplingMode));
      if (inSamplingMode) {
        // woodcock:
        float t        = segment.range.lower;
        float majorant = segment.majorant;
        // LCG<4> &rand = (LCG<4> &)rngSeed;
        while (true) {
          float dt = - logf(1.f-rand())/majorant;
          t += dt;
          if (t >= segment.range.upper)
          break;

          vec3f P = org+t*dir;
          float f = findSample(&self,
                               self.nodes,segment.node,P
                               // ,ray.dbg
                               );
          if (isnan(f))
          continue;
          
          vec4f sample = self.xf.map(f);
          // if (dbg)
          //   printf("DID find sample %f at %f, mapped %f %f %f: %f\n",
          //          f,t,sample.x,sample.y,sample.z,sample.w);
          if (sample.w >= rand()*majorant) {
            hit_t = t;
            ray.setVolumeHit(P,t,getPos(sample));
            break;
          }
        }
      } else {
        // leaf, but NOT sampling mode
        range1f range = segment.range;
        if (hit_t < range.upper)
          printf("RANGE TOO FAR\n");
        // segment.range.upper = min(segment.range.upper,hit_t);
        hit_t = intersectLeaf(ray,range,self,
                              segment.node.offset,
                              segment.node.offset+segment.node.count
                              // ,dbg
                              );
      }
      haveValidSegment = false;
    }
  

      
    // #if AWT_SAMPLE_MODE
      
    //       if (segment.node.count) {
    //         // optoin 'b' - do intersection on the leaf
    //         // if (dbg)
        
    // // #if PRINT_BALLOT
    // //         float expected_dt = - logf(.5f)/((float)segment.majorant);
    // //         float expectedNumSteps = (segment.range.upper-segment.range.lower) / expected_dt;
    // //         if (ray.dbg)
    // //           printf("LEAF, expected num steps %f\n",expectedNumSteps);
    // // #endif

    //         //   printf("LEAF %i cnt %i\n",segment.node.offset,segment.node.count);
    // #if AWT_SAMPLE_MODE
    //         float leaf_t = intersectLeaf(ray,samples,
    //                                      segment.inSampleMode,
    //                                      segment.majorant,
    //                                      segment.range,self,
    //                                      segment.node.offset,
    //                                      segment.node.offset+segment.node.count);
    // #else
    //         float leaf_t = intersectLeaf(ray,samples,segment.range,
    //                                      self,
    //                                      segment.node.offset,
    //                                      segment.node.offset+segment.node.count);
    // #endif
    //         hit_t = min(hit_t,leaf_t);
    //         haveValidSegment = false;
    //         // if (dbg) printf("new t %f leaf %f\n",hit_t,leaf_t);
    //       } else {
    //         // option 'c' - we're sampling .... not yet implemented
    //         printf("sampling!?\n");
    //       }
  
    //
    //
    // TODO: if expected num steps is small enough, just sample
    // isntead of doing per-element intersection
    //
  
    if (hit_t < optixGetRayTmax())  {
      // ray.hadHit = true;
      // ray.tMax = hit_t;
      // ray.hit.P = ray.org + hit_t * ray.dir;
      // ray.hit.N = vec3f(0.f);
      // ray.setVolumeHit(ray.org + hit_t * ray.dir,hit_t,
      optixReportIntersection(hit_t, 0);
    }
  }
}
}
