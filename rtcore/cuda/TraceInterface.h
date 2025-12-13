// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

// #include <owl/owl.h>
#include "rtcore/cudaCommon/ComputeInterface.h"
#include "rtcore/cuda/TraceKernel.h"
#include "rtcore/cuda/Group.h"

#include "rtcore/cuda/ProgramInterface.h"
#include "rtcore/cudaCommon/Device.h"

#include <cuBQL/bvh.h>
#include <cuBQL/math/Ray.h>
#include <cuBQL/queries/triangleData/Triangle.h>
#include <cuBQL/queries/triangleData/math/rayTriangleIntersections.h>
#include <cuBQL/traversal/rayQueries.h>

namespace rtc {
  namespace cuda {

#if BARNEY_CUBQL_BVH_WIDTH == 2
    using bvh_t = cuBQL::bvh3f;
#else
    using bvh_t = cuBQL::WideBVH<float,BARNEY_CUBQL_BVH_WIDTH>;
#endif
    
#if 1
    inline __device__
    void TraceInterface::traceRay(rtc::AccelHandle _world,
                                  vec3f org,
                                  vec3f dir,
                                  float t0,
                                  float t1,
                                  void *prdPtr)
    {
      using Triangle3f = cuBQL::triangle_t<float>;
      using RayTriangleIntersection = cuBQL::RayTriangleIntersection_t<float>;
      
      if (fabsf(dir.x) < 1e-6f) dir.x = 1e-6f;
      if (fabsf(dir.y) < 1e-6f) dir.y = 1e-6f;
      if (fabsf(dir.z) < 1e-6f) dir.z = 1e-6f;

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
      
      ::cuBQL::ray3f ray((const cuBQL::vec3f&)world.org,
                         (const cuBQL::vec3f&)world.dir,
                         t0,t1);
      auto intersectPrim = [&](uint32_t primIdx) -> float
      {
        GeomGroup::DeviceRecord *group 
          = (GeomGroup::DeviceRecord *)&currentInstance->group;
        GeomGroup::Prim *prims = group->prims;
        GeomGroup::Prim prim = prims[primIdx];
        current.primID = prim.primID;
        current.geomID = prim.geomID;
        current.tMax = accepted.tMax;
        uint8_t *geomSBT = group->sbt + prim.geomID  * group->sbtEntrySize;
        Geom::SBTHeader *header
          = (Geom::SBTHeader *)geomSBT;
        this->geomData = (header+1);
        if (group->isTrianglesGroup) {
          vec3i indices = header->triangles.indices[prim.primID];
          Triangle3f tri{
            (const cuBQL::vec3f&)header->triangles.vertices[indices.x],
            (const cuBQL::vec3f&)header->triangles.vertices[indices.y],
            (const cuBQL::vec3f&)header->triangles.vertices[indices.z]
          };
          RayTriangleIntersection isec;
          if (!isec.compute(ray,tri)) 
            return accepted.tMax;
        } else {
          header->user.intersect(*this);
          if (current.tMax >= accepted.tMax)
            return accepted.tMax;
        }
        rejectThisHit = false;
        if (header->ah) header->ah(*this);
        if (!rejectThisHit) {
          accepted.tMax = current.tMax;
          accepted.primID = current.primID;
          accepted.geomID = current.geomID;
          accepted.instID = current.instID;
          accepted.triangleBarycentrics = current.triangleBarycentrics;
          acceptedSBT = header;
        }
        return ray.tMax;
      };
      auto enterBlas = [this,model]
        (cuBQL::ray3f &ray,
         int instID) -> ::cuBQL::bvh3f
      {
        current.instID  = instID;
        currentInstance = model->instanceRecords+current.instID;
        (vec3f&)ray.origin
          = xfmPoint(currentInstance->worldToObjectXfm,world.org);
        (vec3f&)ray.direction
          = xfmVector(currentInstance->worldToObjectXfm,world.dir);
        
        bvh3f blas = {0,0,0,0};
        blas.nodes = currentInstance->group.bvhNodes;
        return blas;
      };
      auto leaveBlas = [this](cuBQL::ray3f &ray) -> void {
        ray.origin    = (const cuBQL::vec3f&)world.org;
        ray.direction = (const cuBQL::vec3f&)world.dir;
      };
      
      ::cuBQL::shrinkingRayQuery::twoLevel::forEachPrim
          (enterBlas,leaveBlas,intersectPrim,model->bvh,ray);
      
      if (acceptedSBT && acceptedSBT->ch) {
        current = accepted;
        this->geomData = (acceptedSBT+1);
        acceptedSBT->ch(*this);
      }
    }
#else
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
      t0 = max(t0,reduce_max(nr));
      t1 = min(t1,reduce_min(fr));
      return t0 <= t1;
    }



    inline __device__
    void TraceInterface::traceRay(rtc::AccelHandle _world,
                                  vec3f org,
                                  vec3f dir,
                                  float t0,
                                  float t1,
                                  void *prdPtr)
    {
      if (fabsf(dir.x) < 1e-6f) dir.x = 1e-6f;
      if (fabsf(dir.y) < 1e-6f) dir.y = 1e-6f;
      if (fabsf(dir.z) < 1e-6f) dir.z = 1e-6f;
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
      if (!nodes) {
        // this node seems to not have any content
        return;
      }
      int nodeID = 0;
      float node_t0=tMin, node_t1 = accepted.tMax;
      if (!boxTest(node_t0,node_t1,nodes[0].bounds,org,dir)) {
        return;
      }

      bool done = false;
      bool inTopLevel = true;
      while (true) {
        while (true) {
          while (nodeID == -1) {
            if (stackPtr == topLevelStackBase) {
              done = true;
              break;
            }
            
            if (stackPtr == stackBase) {
              // going back to parent
              org = world.org;
              dir = world.dir;
              currentInstance = 0;
              nodes = model->bvh.nodes;
              stackBase = topLevelStackBase;
              currentInstance = 0;
              inTopLevel = true;
            }
            --stackPtr;
            if (stackPtr->dist > accepted.tMax)
              continue;
            nodeID = stackPtr->node;
          }
          if (done) break;

          const bvh3f::Node *node = &nodes[nodeID];          
          if (node->admin.count == 0) {
            uint32_t child = node->admin.offset;
            float near0=tMin;
            float near1=tMin;
            float far0 =accepted.tMax;
            float far1 =accepted.tMax;
            nodeID = -1;
            // if (dbg)
            //   printf("checking child 0 (# %i)...\n",child+0);
            boxTest(near0,far0,nodes[child+0].bounds,org,dir);
            // if (dbg)
            //   printf("-> %f %f, %s\n",near0,far0,near0<far0?"HIT":"miss");
            // if (dbg)
            //   printf("checking child 1 (# %i...\n",child+1);
            boxTest(near1,far1,nodes[child+1].bounds,org,dir);
            // if (dbg)
            //   printf("-> %f %f %s\n",near1,far1,near1<far1?"HIT":"miss");
            
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
                ++stackPtr;
                if ((stackPtr - topLevelStackBase) >= STACK_DEPTH) {
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
          } else {
            break;
          }
        }
        if (done)
          break;

        GeomGroup::DeviceRecord *group 
          = (GeomGroup::DeviceRecord *)&currentInstance->group;
        GeomGroup::Prim *prims = group->prims;
        const bvh3f::Node *node = nodes+nodeID;
        for (int primNo=0;primNo<node->admin.count;primNo++) {
          GeomGroup::Prim prim = prims[node->admin.offset+primNo];
          current.primID = prim.primID;
          current.geomID = prim.geomID;
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
          } else {
            header->user.intersect(*this);
            if (current.tMax >= accepted.tMax)
              continue;
          }
          rejectThisHit = false;
          if (header->ah) header->ah(*this);
          if (!rejectThisHit) {
            accepted.tMax = current.tMax;
            accepted.primID = current.primID;
            accepted.geomID = current.geomID;
            accepted.instID = current.instID;
            accepted.triangleBarycentrics = current.triangleBarycentrics;
            acceptedSBT = header;
          }
        }
        nodeID = -1;
      }
      if (acceptedSBT && acceptedSBT->ch) {
        current = accepted;
        this->geomData = (acceptedSBT+1);
        acceptedSBT->ch(*this);
      }
    }
#endif
    
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

#if RTC_DEVICE_CODE
# define RTC_CUDA_TRACEKERNEL(name,Class)                       \
  __global__                                                    \
  void rtc_cuda_run_##name(::rtc::cuda::TraceInterface ti)      \
  {                                                             \
    Class::run(ti);                                             \
  }                                                             
#else
# define RTC_CUDA_TRACEKERNEL(name,Class)                       \
  __global__                                                    \
  void rtc_cuda_run_##name(::rtc::cuda::TraceInterface ti);
#endif

#define RTC_EXPORT_TRACE2D(name,Class)                          \
  RTC_CUDA_TRACEKERNEL(name,Class)                              \
  void rtc_cuda_launch_##name(rtc::Device *device,              \
                              vec2i dims,                       \
                              const void *lpData)               \
  {                                                             \
    vec2i bs(16,16);                                            \
    vec2i nb = divRoundUp(dims,bs);                             \
    ::rtc::cuda::TraceInterface ti;                             \
    ti.lpData = lpData;                                         \
    rtc_cuda_run_##name                                         \
      <<<dim3{(unsigned)nb.x,(unsigned)nb.y,(unsigned)1},       \
      dim3{(unsigned)bs.x,(unsigned)bs.y,(unsigned)1},          \
      0,device->stream>>>(ti);                                  \
  }                                                             \
  


