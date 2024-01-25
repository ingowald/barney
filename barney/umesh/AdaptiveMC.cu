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

#include "barney/umesh/AdaptiveMC.h"
#include "barney/umesh/os/ObjectSpace-common.h"

namespace barney {

  void reallocNodes(AdaptiveMC::Forest *forest, int newSize)
  {
    AdaptiveMC::Node *newNodes = 0;
    BARNEY_CUDA_CALL(Malloc((void**)&newNodes,newSize*sizeof(*newNodes)));
    BARNEY_CUDA_CALL(Memcpy(newNodes,forest->nodes,
                            forest->numNodes*sizeof(*newNodes),
                            cudaMemcpyDefault));

    if (forest->nodes)
      BARNEY_CUDA_CALL(Free(forest->nodes));
    forest->nodes = newNodes;
    forest->numNodes = newSize;
  }

  void reallocLeaves(AdaptiveMC::Forest *forest, int newSize)
  {
    if (forest->leaves)
      BARNEY_CUDA_CALL(Free(forest->leaves));
    BARNEY_CUDA_CALL(Malloc((void**)&forest->leaves,newSize*sizeof(*forest->leaves)));
    forest->numLeaves = newSize;
  }

  __global__ void d_clearLeaves(AdaptiveMC::Forest *forest)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= forest->numLeaves) return;
    
    auto &leaf = forest->leaves[tid];
    leaf.duringBuild.clear();
  }

  __global__ void d_initRoots(AdaptiveMC::Forest *forest, int numRoots)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numRoots) return;
    AdaptiveMC::Node root;
    root.isLeaf = true;
    root.offset = tid;
    root.cellID.x = tid % forest->rootDims.x;
    root.cellID.y = (tid / forest->rootDims.x) % forest->rootDims.y;
    root.cellID.z = tid / (forest->rootDims.x * forest->rootDims.y);
    root.level = 0;
    forest->nodes[tid]  = root;

    auto &leaf = forest->leaves[tid];
    leaf.cellID = root.cellID;
    leaf.level  = root.level;
    // AdaptiveMC::Leaf leaf;
    // forest->leaves[tid] = leaf;
  }
  
  void clearLeaves(AdaptiveMC::Forest *forest)
  {
    d_clearLeaves<<<divRoundUp((int)forest->numNodes,128),128>>>
      (forest);
  }

  inline __both__ vec3i operator<<(vec3i v, int s)
  { return { v.x<<s, v.y<<s, v.z<<s }; }

  inline __device__
  bool allOutside(Plane p, box3f bb)
  {
    vec3f furthestCorner
      (p.N.x > 0 ? bb.upper.x : bb.lower.x,
       p.N.y > 0 ? bb.upper.y : bb.lower.y,
       p.N.z > 0 ? bb.upper.z : bb.lower.z);
    if (p.eval(furthestCorner) < 0.f)
      return true;
    return false;
  }
  
  inline __device__
  bool overlaps(box3f a, box3f b)
  {
    return !(a.lower.x > b.upper.x |
             a.lower.y > b.upper.y |
             a.lower.z > b.upper.z |
             b.lower.x > a.upper.x |
             b.lower.y > a.upper.y |
             b.lower.z > a.upper.z);
  }

  inline __device__
  bool potentiallyOverlaps(vec4i tet, const float4 *vertices, box3f cellBounds)
  {
    float4 __v0 = vertices[tet.x];
    float4 __v1 = vertices[tet.y];
    float4 __v2 = vertices[tet.z];
    float4 __v3 = vertices[tet.w];
    vec3f v0 = getPos(__v0);
    vec3f v1 = getPos(__v1);
    vec3f v2 = getPos(__v2);
    vec3f v3 = getPos(__v3);
    
    Plane p0, p1, p2, p3;
    
    p3.set(v0,v1,v2);
    p2.set(v0,v3,v1);
    p1.set(v0,v2,v3);
    p0.set(v1,v3,v2);

    if (allOutside(p0,cellBounds)) return false;
    if (allOutside(p1,cellBounds)) return false;
    if (allOutside(p2,cellBounds)) return false;
    if (allOutside(p3,cellBounds)) return false;

    return true;
  }
  
  inline __device__
  bool potentiallyOverlaps(AdaptiveMC::Forest *forest,
                AdaptiveMC::Node node,
                UMeshField::DD mesh,
                Element elt,
                box3f eltBounds)
  {
    vec3i intCoords = node.cellID << node.level;
    box3f cellBounds;
    vec3f cellWidth = 1.f/(1<<node.level);
    // RELATIVE
    cellBounds.lower = vec3f(intCoords) * cellWidth;
    cellBounds.upper = cellBounds.lower + cellWidth;

    vec3f blockWidth = forest->worldBounds.size()*rcp(vec3f(forest->rootDims));
    cellBounds.lower = forest->worldBounds.lower + blockWidth * cellBounds.lower;
    cellBounds.upper = forest->worldBounds.lower + blockWidth * cellBounds.upper;

    if (!overlaps(cellBounds,eltBounds))
      return false;
    if (elt.type == Element::TET)
      return potentiallyOverlaps(mesh.tetIndices[elt.ID],mesh.vertices,cellBounds);
    else
      return true;
  }
  
  struct StackEntry {
    int nodeID;
    int numMore;
  };
  
  inline __device__
  void rasterElement(AdaptiveMC::Forest *forest,
                     UMeshField::DD mesh,
                     Element elt,
                     // box4f cellsBB,
                     StackEntry *stackBase,
                     StackEntry *stackPtr)
  {
    box3f bb = getBox(mesh.eltBounds(elt));
    while (true) {
      if (stackPtr == stackBase) return;

      StackEntry job = *--stackPtr;
      if (job.numMore) 
        *stackPtr++ = { job.nodeID+1,job.numMore-1 };

      AdaptiveMC::Node node = forest->nodes[job.nodeID];
      if (!potentiallyOverlaps(forest,node,mesh,elt,bb))
        continue;

      if (node.isLeaf) {
        auto &leaf = forest->leaves[node.offset];
        atomicAdd(&leaf.duringBuild.numElements,1);
        atomicAdd(&leaf.duringBuild.sumEdgeLengths,sqrtf(length(bb.size()))*(1<<leaf.level));
      } else {
        *stackPtr++ = { (int)node.offset, 2*2*2 - 1 };
      }
    }
  }
                     
  __global__
  void d_rasterMesh(AdaptiveMC::Forest *forest, UMeshField::DD mesh)
  {
    StackEntry stackBase[32];
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= mesh.numElements) return;

    Element elt = mesh.elements[tid];
    box4f bb = mesh.eltBounds(elt);
    (vec3f&)bb.lower = ((vec3f&)bb.lower - forest->worldBounds.lower) * rcp(forest->worldBounds.size());
    (vec3f&)bb.upper = ((vec3f&)bb.upper - forest->worldBounds.lower) * rcp(forest->worldBounds.size());
    (vec3f&)bb.lower = (vec3f&)bb.lower * vec3f(forest->rootDims);
    (vec3f&)bb.upper = (vec3f&)bb.upper * vec3f(forest->rootDims);
    vec3i lo = vec3i((vec3f&)bb.lower);
    vec3i hi = min(vec3i((vec3f&)bb.upper),forest->rootDims-1);

    for (int iz=lo.z;iz<=hi.z;iz++)
      for (int iy=lo.y;iy<=hi.y;iy++)
        for (int ix=lo.x;ix<=hi.x;ix++) {
          int rootID = ix + forest->rootDims.x*(iy + forest->rootDims.y*(iz));
          StackEntry *stackPtr = stackBase;
          *stackPtr++ = { rootID, 0 };

          rasterElement(forest,mesh,elt// ,bb
                        ,stackBase,stackPtr);
        }
    
  }

  void rasterMesh(AdaptiveMC::Forest *forest, UMeshField::SP mesh)
  {
    d_rasterMesh<<<divRoundUp((int)mesh->elements.size(),128),128>>>
      (forest,mesh->getDD(0));
  }

  void sortOrder(AdaptiveMC::NextSplitCell *splitOrder, int num)
  {
    std::sort(&splitOrder->bits,&splitOrder->bits+num);
  }
  
  __global__
  void d_buildOrder(AdaptiveMC::Forest *forest,
                    AdaptiveMC::NextSplitCell *splitOrder,
                    int numLeaves)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numLeaves) return;

    auto &order = splitOrder[tid];
    auto &leaf = forest->leaves[tid].duringBuild;
    order.cell = tid;
    order.priority
      = leaf.numElements == 0
      ? 1e20f
      : float(leaf.sumEdgeLengths / leaf.numElements);
  }
  
  void buildOrder(AdaptiveMC::Forest *forest,
                  AdaptiveMC::NextSplitCell *splitOrder)
  {
    int num = forest->numLeaves;
    d_buildOrder<<<divRoundUp(num,128),128>>>
      (forest, splitOrder, num);
  }
  
  void AdaptiveMC::build(UMeshField::SP mesh)
  {
    Forest *forest = 0;
    BARNEY_CUDA_CALL(MallocManaged((void**)&forest,sizeof(*forest)));

    forest->worldBounds = mesh->worldBounds;

    float maxWidth = reduce_max(getBox(mesh->worldBounds).size());
    int ROOT_GRID_SIZE
      = 16 + int(sqrtf(mesh->elements.size())/100);
    
    vec3i rootDims = 1+vec3i(getBox(mesh->worldBounds).size() * ((ROOT_GRID_SIZE-1) / maxWidth));
    *forest = Forest();
    forest->rootDims = rootDims;

    int numRoots = rootDims.x*rootDims.y*rootDims.z;
    reallocNodes(forest,numRoots);
    reallocLeaves(forest,numRoots);

    d_initRoots<<<divRoundUp(numRoots,128),128>>>
      (forest,numRoots);
    // BARNEY_CUDA_CALL((void**)&forest->nodes,
    //                  forest->numNodes*sizeof(int));

    PING;
    while (true) {
      clearLeaves(forest);
      rasterMesh(forest,mesh);
      NextSplitCell *splitOrder = 0;
      BARNEY_CUDA_CALL(MallocManaged((void**)&splitOrder,forest->numLeaves*sizeof(*splitOrder)));
      buildOrder(forest,splitOrder);
      sortOrder(splitOrder,forest->numLeaves);
      BARNEY_CUDA_CALL(Free(splitOrder));
      // split nodes
      // resize
    }
  }
  
}



