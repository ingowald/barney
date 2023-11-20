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
#include "barney/umesh/AWT.h"

namespace barney {
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
    //     bounds.extend(self.mesh.eltBounds(self.mesh.elements[i]));
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
    // float majorant = cluster.majorant;
    
    // ray.hadHit = true;
    // ray.color = .8f;//owl::randomColor(primID);
    // ray.primID = primID;
    ray.tMax = optixGetRayTmax();

    vec3f P = ray.org + ray.tMax * ray.dir;
    // if (ray.dbg)
    //   printf("CENTRAL DIFF prim %i at %f %f %f\n",
    //          primID,
    //          P.x,P.y,P.z);

    vec3f N = normalize(ray.dir);
    ray.hadHit = 1;
    ray.hit.N = N;
    ray.hit.P = P;
    // ray.hit.baseColor = randomColor(primID);
  }
  
  struct __barney_align(16) AWTSegment {
    AWTNode::NodeRef node;
    float  majorant;
    range1f range;
  };
    
  inline __device__ void orderSegments(AWTSegment &a,
                                       AWTSegment &b)
  {
    if (a.range.lower < b.range.lower) {
      AWTSegment c = a;
      a = b;
      b = c;
    }
  }
  
  OPTIX_INTERSECT_PROGRAM(UMeshAWTIsec)()
  {
    const int primID = optixGetPrimitiveIndex();
    const auto &self
      = owl::getProgramData<typename UMeshAWT::DD>();
    auto &ray
      = owl::getPRD<Ray>();
    // bool dbg = ray.dbg;
    
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
    float hit_t = segment.range.upper;
    vec3f org = ray.org;
    vec3f dir = ray.dir;
    bool isHittingTheBox
      = boxTest(segment.range.lower,segment.range.upper,bounds,org,dir);

    if (!isHittingTheBox) {
      return;
    }
    
    AWTSegment stackBase[32];
    AWTSegment *stackPtr = stackBase;
    // AWTSegment *stackHi = stackBase+32;
    while (true) {
      // if (dbg) printf("---------------------- starting trav ofs %i cnt %i range %f %f\n",
      //                 segment.node.offset,segment.node.count,
      //                 segment.range.lower,segment.range.upper);
      while (segment.node.count == 0) {
        // if (dbg) printf("* --------- inner %i range %f %f\n",
        //                 segment.node.offset,
        //                 segment.range.lower,segment.range.upper);
        auto node = self.nodes[segment.node.offset];

        AWTSegment childSeg[4];
#pragma unroll(4)
        for (int c=0;c<4;c++) {
          float tt0 = segment.range.lower;
          float tt1 = segment.range.upper;
          childSeg[c] = { node.child[c], 
                          node.majorant[c],
                          range1f{ INFINITY, INFINITY } };
          if (node.depth[c] == -1) {
            // if (dbg) printf("** child %i INVALID\n",c);
          } else if (node.majorant[c] == 0.f) {
            // if (dbg) printf("** child %i zero majorant...\n",c);
          } else if (!boxTest(tt0,tt1,node.bounds[c],org,dir)) {
            auto box = node.bounds[c];
            // if (dbg) printf("** child %i miss... box was (%f %f %f)(%f %f %f)\n",c,
            //                 box.lower.x,
            //                 box.lower.y,
            //                 box.lower.z,
            //                 box.upper.x,
            //                 box.upper.y,
            //                 box.upper.z
            //                 );
          } else {
            childSeg[c].range = range1f{ tt0, tt1 };
            // if (dbg) printf("** child %i range %f %f\n",
            //                 c,
            //                 tt0,tt1);
          }
        }

        orderSegments(childSeg[0],childSeg[1]);
        orderSegments(childSeg[1],childSeg[2]);
        orderSegments(childSeg[2],childSeg[3]);

        orderSegments(childSeg[0],childSeg[1]);
        orderSegments(childSeg[1],childSeg[2]);

        orderSegments(childSeg[0],childSeg[1]);

        // if (dbg) {
        //   for (int c=0;c<4;c++)
        //     printf("seg %i range %f %f\n",c,
        //            childSeg[c].range.lower,
        //            childSeg[c].range.upper);
        // }

        if (childSeg[0].range.lower < hit_t) *stackPtr++ = childSeg[0];
        // if (stackPtr >= stackHi) { printf("stack overflow\n"); return; }
        if (childSeg[1].range.lower < hit_t) *stackPtr++ = childSeg[1];
        // if (stackPtr >= stackHi) { printf("stack overflow\n"); return; }
        if (childSeg[2].range.lower < hit_t) *stackPtr++ = childSeg[2];
        // if (stackPtr >= stackHi) { printf("stack overflow\n"); return; }
#if 1
        if (childSeg[3].range.lower >= hit_t) {
          bool foundOneToPop = false;
          while (stackPtr > stackBase) {
            segment = *--stackPtr;
            if (segment.range.lower >= hit_t)
              continue;
            segment.range.upper = min(segment.range.upper,hit_t);
            foundOneToPop = true;
            break;
          }
          if (!foundOneToPop) {
            segment.node.count = 0;
            break;
          }
        } else
          segment = childSeg[3];
          
#else
        if (childSeg[3].range.lower >= hit_t)
          break;
        segment = childSeg[3];
#endif
      }
      // check if valid leaf, and if so, intersect
      if (segment.node.count) {
        // if (dbg)
        //   printf("LEAF %i cnt %i\n",segment.node.offset,segment.node.count);
        float leaf_t = intersectLeaf(ray,segment.range,self,
                                     segment.node.offset,
                                     segment.node.offset+segment.node.count);
        hit_t = min(hit_t,leaf_t);
        // if (dbg) printf("new t %f leaf %f\n",hit_t,leaf_t);
      }
      
      //pop:
      bool foundOneToPop = false;
      while (stackPtr > stackBase) {
        segment = *--stackPtr;
        if (segment.range.lower >= hit_t)
          continue;
        segment.range.upper = min(segment.range.upper,hit_t);
        foundOneToPop = true;
        break;
      }

      if (!foundOneToPop)
        break;
    }

    //
    //
    // TODO: if expected num steps is small enough, just sample
    // isntead of doing per-element intersection
    //

    if (hit_t < optixGetRayTmax())  {
      optixReportIntersection(hit_t, 0);
    }
  }
}
