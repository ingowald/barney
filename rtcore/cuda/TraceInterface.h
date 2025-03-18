// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

#include <owl/owl.h>
#include "rtcore/cudaCommon/ComputeInterface.h"
#include "rtcore/cuda/TraceKernel.h"

#if !BARNEY_DEVICE_PROGRAM
# error "RTcore.h should only ever be included by device programs"
#endif

namespace rtc {
  namespace cuda {

    using namespace rtc::cuda_common;

    struct Instance {
      affine3f objectToWorldXfm;
      affine3f worldToObjectXfm;
    };
    
    /*! the interface that pipeline programs use to talk to / query
      data from the ray tracing core */
    inline __device__ const void *getLaunchParamsPointer();
    struct TraceInterface {
      inline __device__ void ignoreIntersection() 
      { rejectThisHit = true; }
      
      inline __device__ void reportIntersection(float t, int i) 
      { current.tMax = t; }

      inline __device__ void *getPRD() const
      { return prd; }
      
      inline __device__ const void *getProgramData() const
      { return geomData; }
      
      inline __device__ const void *getLPData() const
      { return lpData; }
      
      inline __device__ vec3i getLaunchDims()  const
      {
        return (const vec3i&)blockDim * (const vec3i &)gridDim;
      }
      
      inline __device__ vec3i getLaunchIndex() const
      { return (const vec3i&)threadIdx + (const vec3i&)blockIdx*(const vec3i&)blockDim; }

      inline __device__ vec2f getTriangleBarycentrics() const
      { return current.triangleBarycentrics; }
      
      inline __device__ int getPrimitiveIndex() const
      { return current.primID; }
      
      inline __device__ int getInstanceIndex() const
      { return current.instID; }
      
      inline __device__ float getRayTmax() const
      { return current.tMax; }
      
      inline __device__ float getRayTmin() const
      { return tMin; }

      inline __device__ vec3f getObjectRayDirection() const
      { return object.dir; }
      
      inline __device__ vec3f getObjectRayOrigin() const
      { return object.org; }
      
      inline __device__ vec3f getWorldRayDirection() const
      { return world.dir; }
      
      inline __device__ vec3f getWorldRayOrigin() const
      { return world.org; }
      
      inline __device__
      vec3f transformNormalFromObjectToWorldSpace(vec3f v) const;
      
      inline __device__
      vec3f transformPointFromObjectToWorldSpace(vec3f v) const;
      
      inline __device__
      vec3f transformVectorFromObjectToWorldSpace(vec3f v) const;
      
      inline __device__
      vec3f transformNormalFromWorldToObjectSpace(vec3f v) const;
      
      inline __device__
      vec3f transformPointFromWorldToObjectSpace(vec3f v) const;
      
      inline __device__
      vec3f transformVectorFromWorldToObjectSpace(vec3f v) const;
      
      inline __device__
      void traceRay(rtc::device::AccelHandle world,
                    vec3f org,
                    vec3f dir,
                    float t0,
                    float t1,
                    void *prdPtr);
      
      inline __device__
      bool intersectTriangle(const vec3f v0,const vec3f v1,const vec3f v2);
      
      // launch params
      const void  *lpData;
      
      // ray/traversal state:
      void  *prd;
      void  *geomData;
      float  tMin;
      Geom::SBTHeader *acceptedSBT;
      struct {
        vec2f  triangleBarycentrics;
        int    primID;
        int    instID;
        float  tMax;
      } current, accepted;
      struct {
        vec3f org;
        vec3f dir;
      } world, object;
      const InstanceGroup::InstanceRecord *currentInstance;
      bool rejectThisHit;
    };


    inline __device__
    bool TraceInterface::intersectTriangle(const vec3f v0,
                                           const vec3f v1,
                                           const vec3f v2)
    {
      vec3f e1 = v1-v0;
      vec3f e2 = v2-v0;

      vec3f N = cross(e1,e2);
      if (fabsf(dot(object.dir,N)) < 1e-12f) return false;

      // P = o+td
      // dot(P-v0,N) = 0
      // dot(o+td-v0,N) = 0
      // dot(td,N)+dot(o-v0,N)=0
      // t*dot(d,N) = -dot(o-v0,N)
      // t = -dot(o-v0,N)/dot(d,N)
      float t = -dot(object.org-v0,N)/dot(object.dir,N);

      // printf("ISEC t %f in [%f %f]\n",t,tMin,current.tMax);
      
      if (t <= 0.f || t >= current.tMax) return false;

      vec3f P = object.org - v0 + t*object.dir;

      float e1u,e2u,Pu;
      float e1v,e2v,Pv;
      if (fabsf(N.x) >= max(fabsf(N.y),fabsf(N.z))) {
        e1u = e1.y; e2u = e2.y; Pu = P.y;
        e1v = e1.z; e2v = e2.z; Pv = P.z;
      } else if (fabsf(N.y) > fabsf(N.z)) {
        e1u = e1.x; e2u = e2.x; Pu = P.x;
        e1v = e1.z; e2v = e2.z; Pv = P.z;
      } else {
        e1u = e1.x; e2u = e2.x; Pu = P.x;
        e1v = e1.y; e2v = e2.y; Pv = P.y;
      }
      auto det = [](float a, float b, float c, float d) -> float
      { return a*d - c*b; };
      
      // P = v0 + u * e1 + v * e2 + h * N
      // (P-v0) = [e1,e2]*(u,v,h)
      if (det(e1u,e1v,e2u,e2v) == 0.f) return false;

#if 1
      float u = det(Pu,e2u,Pv,e2v)/det(e1u,e2u,e1v,e2v);
      float v = det(e1u,Pu,e1v,Pv)/det(e1u,e2u,e1v,e2v);
#else
      float u = det(Pu,Pv,e2u,e2v)/det(e1u,e1v,e2u,e2v);
      float v = det(e1u,e1v,Pu,Pv)/det(e1u,e1v,e2u,e2v);
#endif
      // printf("ISEC uv %f %f\n",u,v);
      if ((u < 0.f) || (v < 0.f) || ((u+v) >= 1.f)) return false;

      current.triangleBarycentrics = vec2f{ u,v };
      current.tMax = t;
      return true;
    }

    inline __device__
    bool boxTest(float &t0, float &t1,
                 const cuBQL::box3f bb,
                 vec3f org,
                 vec3f dir)
    {
      vec3f lo = ((const vec3f &)bb.lower - org) * rcp(dir);
      vec3f hi = ((const vec3f &)bb.upper - org) * rcp(dir);
      vec3f nr = min(lo,hi);
      vec3f fr = max(lo,hi);
#if 0
      printf("box %f %f %f : %f %f %f\n",
             bb.lower.x,
             bb.lower.y,
             bb.lower.z,
             bb.upper.x,
             bb.upper.y,
             bb.upper.z);
      printf("t0 %f (%f %f %f) t1  %f (%f %f %f)\n",
             t0,nr.x,nr.y,nr.z,
             t1,fr.x,fr.y,fr.z);
#endif
      t0 = max(t0,reduce_max(nr));
      t1 = min(t1,reduce_min(fr));
      return t0 <= t1;
    }
    
    inline __device__
    void TraceInterface::traceRay(rtc::device::AccelHandle _world,
                                  vec3f org,
                                  vec3f dir,
                                  float t0,
                                  float t1,
                                  void *prdPtr)
    {
            if (fabsf(dir.x) < 1e-6f) dir.x = 1e-6f;
            if (fabsf(dir.y) < 1e-6f) dir.y = 1e-6f;
            if (fabsf(dir.z) < 1e-6f) dir.z = 1e-6f;
            
      bool dbg = false;
      if (t0 < 0.f) {
        dbg= true;
        t0 = 0.f;
      }

      if (dbg)
        printf("================================= TRACING %f %f %f : %f %f %f : %f\n",
               org.x,
               org.y,
               org.z,
               dir.x,
               dir.y,
               dir.z,
               t1);
      struct StackEntry {
        uint32_t node;
        float    dist;
      };
      prd = prdPtr;
      world.org = org;
      world.dir = dir;
      tMin = t0;
      accepted.tMax = t1;
      accepted.primID = -1;
      accepted.instID = -1;
      acceptedSBT = 0;
      InstanceGroup::DeviceRecord *model
        = (InstanceGroup::DeviceRecord *)_world;

      enum { STACK_DEPTH = 64 };
      StackEntry topLevelStackBase[STACK_DEPTH];
      StackEntry *stackBase = topLevelStackBase;
      StackEntry *stackPtr = stackBase;

      const bvh3f::Node *nodes = model->bvh.nodes;
      int nodeID = 0;
      float node_t0=tMin, node_t1 = accepted.tMax;
      if (dbg)
        printf("root bounds (%f %f %f)(%f %f %f)\n",
               nodes->bounds.lower.x,
               nodes->bounds.lower.y,
               nodes->bounds.lower.z,
               nodes->bounds.upper.x,
               nodes->bounds.upper.y,
               nodes->bounds.upper.z);
      if (!boxTest(node_t0,node_t1,nodes[0].bounds,org,dir)) {
        if (dbg)
          printf("MISS root box\n");
        return;
      }

      if (dbg)
        printf("did hit world box. starting trav\n");
      bool done = false;
      bool inTopLevel = true;
      while (true) {
        while (true) {
          while (nodeID == -1) {
            if (dbg)
              printf("!!! popping from stack, stack depth %i %i\n",
                     int(stackPtr-stackBase),int(stackBase-topLevelStackBase));
            if (stackPtr == topLevelStackBase) {
              if (dbg)
                printf(">>>>>>>>>>> hit bottom of stack, done\n");
              done = true;
              break;
            }
            
            if (stackPtr == stackBase) {
              // going back to parent
              if (dbg)
                printf(">>>>>>>>>>> hit bottom of GEOM stack, back to instance stack\n");
              org = world.org;
              dir = world.dir;
              currentInstance = 0;
              nodes = model->bvh.nodes;
              stackBase = topLevelStackBase;
              currentInstance = 0;
              inTopLevel = true;
            }
            --stackPtr;
            if (dbg)
              printf("dist on stack: %f\n",stackPtr->dist);
            if (stackPtr->dist > accepted.tMax)
              continue;
            nodeID = stackPtr->node;
            if (dbg)
              printf("POPPED %i\n",nodeID);
          }
          if (done) break;

          const bvh3f::Node *node = &nodes[nodeID];          
          if (dbg)
            printf("=========== at node %i addr %p, %i:%i\n",
                   nodeID,
                   node,
                   (int)node->admin.offset,
                   (int)node->admin.count);
          // node is a valid node, and we know we've hit its bounds
          if (node->admin.count == 0) {
            uint32_t child = node->admin.offset;
            float near0=tMin;
            float near1=tMin;
            float far0 =accepted.tMax;
            float far1 =accepted.tMax;
            nodeID = -1;
            if (dbg)
              printf("checking child 0 (# %i)...\n",child+0);
            boxTest(near0,far0,nodes[child+0].bounds,org,dir);
            if (dbg)
              printf("-> %f %f, %s\n",near0,far0,near0<far0?"HIT":"miss");
            if (dbg)
              printf("checking child 1 (# %i...\n",child+1);
            boxTest(near1,far1,nodes[child+1].bounds,org,dir);
            if (dbg)
              printf("-> %f %f %s\n",near1,far1,near1<far1?"HIT":"miss");
            
            if (near0 <= far0) {
              nodeID = child+0;
            }
            if (near1 <= far1) {
              if (near0 <= far0) {
                if (near1 <= near0) {
                  stackPtr->dist = near0;
                  stackPtr->node = child+0;
                  nodeID = child+1;
                } else {
                  stackPtr->dist = near1;
                  stackPtr->node = child+1;
                }
                if (dbg)
                  printf("PUSHING %i @ %f\n",stackPtr->node,stackPtr->dist);
                ++stackPtr;
                if ((stackPtr - topLevelStackBase) >= STACK_DEPTH) {
                  printf("STACK OVERFLOW!!!!\n");
                  return;
                }
              } else {
                nodeID = child+1;
              }
            }
            continue;
          }
          if (done) break;
          // leaf - either to or bottom...
          if (inTopLevel) {
            if (node->admin.count != 1)
              printf("MORE THAN ONE INSTANCE!?");
            if (dbg)
              printf("########### hit INSTANCE leaf %p, %i:%i\n",
                     node,
                     (int)node->admin.offset,
                     (int)node->admin.count);
            current.instID = model->bvh.primIDs[node->admin.offset];
            currentInstance = model->instanceRecords+current.instID;
            org = object.org = xfmPoint(currentInstance->worldToObjectXfm,world.org);
            dir = object.dir = xfmVector(currentInstance->worldToObjectXfm,world.dir);
            if (fabsf(dir.x) < 1e-6f) dir.x = 1e-6f;
            if (fabsf(dir.y) < 1e-6f) dir.y = 1e-6f;
            if (fabsf(dir.z) < 1e-6f) dir.z = 1e-6f;
            nodes = currentInstance->group.bvhNodes;
            nodeID = 0;
            stackBase = stackPtr;
            inTopLevel = false;
              
            if (dbg)
              printf(">> NEW RAY %f %f %f : %f %f %f\n",
                     org.x,
                     org.y,
                     org.z,
                     dir.x,
                     dir.y,
                     dir.z);
            
          } else {
            if (dbg)
              printf("########### hit GEOM leaf %i, %i:%i\n",
                     nodeID,
                     (int)node->admin.offset,
                     (int)node->admin.count);
            break;
          }
        }
        if (done)
          break;
        
        // GEOM instance leaf
        GeomGroup::DeviceRecord *group 
          = (GeomGroup::DeviceRecord *)&currentInstance->group;
        GeomGroup::Prim *prims = group->prims;
        const bvh3f::Node *node = nodes+nodeID;
        for (int primNo=0;primNo<node->admin.count;primNo++) {
          GeomGroup::Prim prim = prims[node->admin.offset+primNo];
          current.primID = prim.primID;
          current.tMax = accepted.tMax;
          uint8_t *geomSBT = group->sbt + prim.geomID  * group->sbtEntrySize;
          Geom::SBTHeader *header
            = (Geom::SBTHeader *)geomSBT;
          this->geomData = (header+1);
          if (group->isTrianglesGroup) {
            vec3i indices = header->triangles.indices[prim.primID];
            vec3f v0 = header->triangles.vertices[indices.x];
            vec3f v1 = header->triangles.vertices[indices.y];
            vec3f v2 = header->triangles.vertices[indices.z];
            if (!intersectTriangle(v0,v1,v2))
              continue;
            if (dbg)
              printf("HIT TRI %i @ %f\n",
                     current.primID,current.tMax);
            rejectThisHit = false;
            if (dbg)
              printf("AH is %p\n",header->ah);
            if (header->ah) {
              header->ah(*this);
              if (dbg)
                printf("DONE ah\n");
            }
            if (!rejectThisHit) {
              accepted.tMax = current.tMax;
              accepted.primID = current.primID;
              accepted.instID = current.instID;
              accepted.triangleBarycentrics = current.triangleBarycentrics;
              acceptedSBT = header;
            }
          } else {
            printf("user geom isec not implemented...\n");
          }
        }
        nodeID = -1;
      }
#if 0
                if (dbg)
      printf("accepted ...%p\n",acceptedSBT);
      if (acceptedSBT && acceptedSBT->ch) {
        current = accepted;
        this->geomData = (acceptedSBT+1);
                if (dbg)
        printf("accepted CH ...%p\n",acceptedSBT->ch);
        acceptedSBT->ch(*this);
      }
#endif
                if (dbg)
      printf("done\n");
    }
    
    inline __device__
    vec3f TraceInterface::transformNormalFromObjectToWorldSpace(vec3f v) const
    {
      return xfmVector(currentInstance->objectToWorldXfm.l,
                       (const owl::common::vec3f &)v);
    }

    inline __device__
    vec3f TraceInterface::transformPointFromObjectToWorldSpace(vec3f v) const
    { 
      return xfmPoint(currentInstance->objectToWorldXfm,
                      (const owl::common::vec3f &)v);
    }

    inline __device__
    vec3f TraceInterface::transformVectorFromObjectToWorldSpace(vec3f v) const
    { 
      return xfmVector(currentInstance->objectToWorldXfm.l,
                       (const owl::common::vec3f &)v);
    }

    inline __device__
    vec3f TraceInterface::transformNormalFromWorldToObjectSpace(vec3f v) const
    {
      return xfmVector(currentInstance->worldToObjectXfm.l,
                       (const owl::common::vec3f &)v);
    }

    inline __device__
    vec3f TraceInterface::transformPointFromWorldToObjectSpace(vec3f v) const
    { 
      return xfmPoint(currentInstance->worldToObjectXfm,
                      (const owl::common::vec3f &)v);
    }

    inline __device__
    vec3f TraceInterface::transformVectorFromWorldToObjectSpace(vec3f v) const
    { 
      return xfmVector(currentInstance->worldToObjectXfm.l,
                       (const owl::common::vec3f &)v);
    }
    
  }
}

#define RTC_DECLARE_GLOBALS(ignore) /* ignore */

#define RTC_EXPORT_TRACE2D(name,Class)                          \
  __global__                                                    \
  void rtc_cuda_run_##name(::rtc::cuda::TraceInterface ti)      \
  {                                                             \
    Class::run(ti);                                             \
  }                                                             \
  void rtc_cuda_launch_##name(rtc::Device *device,              \
                              vec2i dims,                       \
                              const void *lpData)               \
  {                                                             \
    vec2i bs(8,8);                                              \
    vec2i nb = divRoundUp(dims,bs);                             \
    ::rtc::cuda::TraceInterface ti;                             \
    ti.lpData = lpData;                                         \
    rtc_cuda_run_##name<<<nb,bs>>>(ti);                         \
  }                                                             \
  
