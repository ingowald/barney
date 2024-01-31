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
#include <fstream>

namespace barney {

  void reallocNodes(AdaptiveMC::Forest *forest, int newSize)
  {
    AdaptiveMC::Node *newNodes = 0;
    BARNEY_CUDA_CALL(MallocManaged((void**)&newNodes,newSize*sizeof(*newNodes)));
    BARNEY_CUDA_CALL(Memcpy(newNodes,forest->nodes,
                            forest->numNodes*sizeof(*newNodes),
                            cudaMemcpyDefault));
    
    if (forest->nodes)
      BARNEY_CUDA_CALL(Free(forest->nodes));
    forest->nodes = newNodes;
    // forest->numNodes = newSize;
  }
  
  void reallocLeaves(AdaptiveMC::Forest *forest, int newSize)
  {
    AdaptiveMC::Leaf *newLeaves = 0;
    BARNEY_CUDA_CALL(MallocManaged((void**)&newLeaves,newSize*sizeof(*forest->leaves)));
    BARNEY_CUDA_CALL(Memcpy(newLeaves,forest->leaves,
                            forest->numLeaves*sizeof(*newLeaves),
                            cudaMemcpyDefault));
    // BARNEY_CUDA_CALL(MallocManaged((void**)&forest->leaves,
    //                                newSize*sizeof(*forest->leaves)));
    // BARNEY_CUDA_CALL(Memcpy(newLeaves,forest->leaves,
    //                         forest->numLeaves*sizeof(*forest->leaves),
    //                         cudaMemcpyDefault));
    if (forest->leaves)
      BARNEY_CUDA_CALL(Free(forest->leaves));
    forest->leaves = newLeaves;
    // forest->numLeaves = newSize;
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
    box3f cellBounds = forest->getCellBounds(node.cellID,node.level);;
    // vec3i intCoords = node.cellID << node.level;
    // vec3f cellWidth = 1.f/(1<<node.level);
    // // RELATIVE
    // cellBounds.lower = vec3f(intCoords) * cellWidth;
    // cellBounds.upper = cellBounds.lower + cellWidth;

    // vec3f blockWidth = forest->rootCellWidth;
    // // vec3f blockWidth = forest->worldBounds.size()*rcp(vec3f(forest->rootDims));
    // cellBounds.lower = forest->worldBounds.lower + blockWidth * cellBounds.lower;
    // cellBounds.upper = forest->worldBounds.lower + blockWidth * cellBounds.upper;

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
                     StackEntry *stackPtr,
                     bool dbg = false)
  {
    box3f bb = getBox(mesh.eltBounds(elt));
    if (dbg) printf("---------------- rastering element (%f %f %f)(%f %f %f)\n",
                    bb.lower.x,
                    bb.lower.y,
                    bb.lower.z,
                    bb.upper.x,
                    bb.upper.y,
                    bb.upper.z);
    while (true) {
      if (stackPtr == stackBase) return;

      StackEntry job = *--stackPtr;
      if (job.numMore) 
        *stackPtr++ = { job.nodeID+1,job.numMore-1 };

      AdaptiveMC::Node node = forest->nodes[job.nodeID];
      box3f nodeBounds = forest->getCellBounds(node.cellID,node.level);
      if (dbg)
        printf("  testing (%i %i %i; %i), (%f %f %f)(%f %f %f)\n",
             node.cellID.x,
             node.cellID.y,
             node.cellID.z,
             node.level,
             nodeBounds.lower.x,
             nodeBounds.lower.y,
             nodeBounds.lower.z,
             nodeBounds.upper.x,
             nodeBounds.upper.y,
             nodeBounds.upper.z);
             
      if (!potentiallyOverlaps(forest,node,mesh,elt,bb))
        continue;

      if (node.isLeaf) {
        auto &leaf = forest->leaves[node.offset];
        if (dbg)
          printf("  - reached leaf %i %i %i; %i\n",
                 leaf.cellID.x,
                 leaf.cellID.y,
                 leaf.cellID.z,
                 leaf.level);
        atomicAdd(&leaf.duringBuild.numElements,1);

        float thisElementLength = length(bb.size());
        thisElementLength = min(thisElementLength,
                                3.f*length(forest->
                                           rootCellWidth)/(1<<leaf.level));
        float thisElementWeight = sqrtf(thisElementLength);//*(1<<leaf.level);
        
        atomicAdd(&leaf.duringBuild.sumEdgeLengths,thisElementWeight);
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

    bool dbg = (tid == 34333);
    for (int iz=lo.z;iz<=hi.z;iz++)
      for (int iy=lo.y;iy<=hi.y;iy++)
        for (int ix=lo.x;ix<=hi.x;ix++) {
          int rootID = ix + forest->rootDims.x*(iy + forest->rootDims.y*(iz));
          if (dbg)
            printf("rastering element into root tree %i\n",rootID);
          StackEntry *stackPtr = stackBase;
          *stackPtr++ = { rootID, 0 };

          rasterElement(forest,mesh,elt// ,bb
                        ,stackBase,stackPtr,
                        dbg);
        }
    
  }

  void rasterMesh(AdaptiveMC::Forest *forest, UMeshField *mesh)
  {
    d_rasterMesh<<<divRoundUp((int)mesh->elements.size(),128),128>>>
      (forest,mesh->getDD(0));
  }

  void sortOrder(AdaptiveMC::Forest *forest,
                 AdaptiveMC::NextSplitJob *splitOrder, int num)
  {
    uint64_t *sortData = (uint64_t *)splitOrder;
    std::sort(sortData,sortData+num);
    for (int i=0;i<min(20,num);i++) {
      auto node = forest->nodes[splitOrder[i].nodeID];
      std::cout << " sort prio " << i << " : " << splitOrder[i].nodeID << "@"
                << node.cellID
                << ";" << node.level << "  prio "
                << splitOrder[i].priority << std::endl;
    }
  }
  
  __global__
  void d_buildOrder(AdaptiveMC::Forest *forest,
                    AdaptiveMC::NextSplitJob *splitOrder)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= forest->numNodes) return;

    int nodeID = tid;
    auto &node = forest->nodes[nodeID];
    if (!node.isLeaf)
      // can only split leaves...
      return;

    int leafID = node.offset;
    auto &order = splitOrder[leafID];
    auto &leaf = forest->leaves[leafID];
    order.nodeID = nodeID;
    order.priority
      = (leaf.duringBuild.numElements == 0)
      ? 1e20f
      : (1.f/leaf.duringBuild.numElements);
      //        1e20f
      // : (float(leaf.duringBuild.sumEdgeLengths)
      //    / float(leaf.duringBuild.numElements)
      //    / (1<<leaf.level));
  }

  void buildOrder(AdaptiveMC::Forest *forest,
                  AdaptiveMC::NextSplitJob *splitOrder)
  {
    int num = forest->numNodes;
    d_buildOrder<<<divRoundUp(num,128),128>>>
      (forest, splitOrder);
  }

  __global__
  void d_splitLeaves(AdaptiveMC::Forest *forest,
                     AdaptiveMC::NextSplitJob *splitOrder,
                     int numToSplit)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numToSplit) return;

    int newLeavesPos = atomicAdd(&forest->numLeaves,7);
    int newNodesPos = atomicAdd(&forest->numNodes,8);

    AdaptiveMC::Node &nodeToSplit = forest->nodes[tid];

    int childID = 0;
    vec3i oldCellID = nodeToSplit.cellID;
    int   oldLevel  = nodeToSplit.level;
    for (int iz=0;iz<2;iz++)
      for (int iy=0;iy<2;iy++)
        for (int ix=0;ix<2;ix++, childID++) {
          vec3i newCellID = 2*oldCellID + vec3i(ix,iy,iz);
          int   newLevel  = oldLevel+1;

          int newLeafIndex
            = (childID==0)
            ? tid
            : (newLeavesPos+childID-1);
          AdaptiveMC::Node &newNode = forest->nodes[newNodesPos+childID];
          AdaptiveMC::Leaf &newLeaf = forest->leaves[newLeafIndex];
          newLeaf.cellID = newCellID;
          newLeaf.level  = newLevel;
          newLeaf.duringBuild.clear();

          newNode.cellID = newCellID;
          newNode.level  = newLevel;
          newNode.offset = newLeafIndex;
          newNode.isLeaf = true;
        }
    nodeToSplit.isLeaf = 0;
    nodeToSplit.offset = newNodesPos;
  }
  
  void splitLeaves(AdaptiveMC::Forest *forest,
                  AdaptiveMC::NextSplitJob *splitOrder)
  {
    int oldNumLeaves = forest->numLeaves;
    int oldNumNodes  = forest->numNodes;

    int numLeavesToSplit = 100+oldNumLeaves/8;
    
    // cannot split more cells than we have:
    numLeavesToSplit = min(numLeavesToSplit,oldNumLeaves);

    // each leaf getting split produced 8 leaves, but that leaf goes
    // away, so the total num *new* leaves produces per split leaf is
    // *7*:
    int newNumLeaves = oldNumLeaves + numLeavesToSplit*(8-1);

    // each newly created leaf also needs a cell to poitn to it; but
    // unlike for leaves the original leaf's cell will not go away
    // (it'll only change type). thus, total num new cells is 8 per
    // leaf being split
    int newNumNodes = oldNumNodes + numLeavesToSplit*8;

    reallocNodes(forest,newNumNodes);
    reallocLeaves(forest,newNumLeaves);
    
    d_splitLeaves<<<divRoundUp(numLeavesToSplit,128),128>>>
      (forest,splitOrder,numLeavesToSplit);
  }
  
  void AdaptiveMC::build(UMeshField *mesh)
  {
    Forest *forest = 0;
    BARNEY_CUDA_CALL(MallocManaged((void**)&forest,sizeof(*forest)));
    *forest = Forest();

    forest->worldBounds = mesh->worldBounds;
    PRINT(forest->worldBounds);
    float maxWidth = reduce_max(getBox(mesh->worldBounds).size());
#if 0
    vec3i rootDims = 1;
#else
    int ROOT_GRID_SIZE = 16 + int(powf(mesh->elements.size(),1.f/3.f)/100);
    
    vec3i rootDims = 1+vec3i(getBox(mesh->worldBounds).size() * ((ROOT_GRID_SIZE-1) / maxWidth));
#endif
    forest->rootDims = rootDims;

    PRINT(forest->rootDims);
    forest->rootCellWidth = forest->worldBounds.size()*rcp(vec3f(forest->rootDims));
    PRINT(forest->rootCellWidth);
    
    int numRoots = rootDims.x*rootDims.y*rootDims.z;
    reallocNodes(forest,numRoots);
    reallocLeaves(forest,numRoots);

    forest->numLeaves = numRoots;
    forest->numNodes  = numRoots;

    d_initRoots<<<divRoundUp(numRoots,128),128>>>
      (forest,numRoots);
    BARNEY_CUDA_SYNC_CHECK();
    // std::cout << "roots:" << std::endl;
    // for (int i=0;i<forest->numLeaves;i++) {
    //   auto leaf = forest->leaves[i];
    //   std::cout << "leaf " << i << " cell " << leaf.cellID
    //             << ";" << leaf.level << "  " << forest->getLeafBounds(leaf)
    //             << std::endl;
    // }
      
    // BARNEY_CUDA_CALL((void**)&forest->nodes,
    //                  forest->numNodes*sizeof(int));

    int targetNumLeaves = 100 * numRoots;
    PING;
    while (forest->numLeaves < targetNumLeaves) {
      PRINT(forest->numLeaves);
      PRINT(forest->numNodes);
      std::cout << "clearing leaves" << std::endl << std::flush;
      clearLeaves(forest);
      BARNEY_CUDA_SYNC_CHECK();

      std::cout << "raster" << std::endl << std::flush;
      rasterMesh(forest,mesh);
      BARNEY_CUDA_SYNC_CHECK();

      NextSplitJob *splitOrder = 0;
      BARNEY_CUDA_CALL(MallocManaged((void**)&splitOrder,forest->numLeaves*sizeof(*splitOrder)));
      BARNEY_CUDA_SYNC_CHECK();

      std::cout << "building order" << std::endl << std::flush;
      buildOrder(forest,splitOrder);
      BARNEY_CUDA_SYNC_CHECK();

      std::cout << "sorting order" << std::endl << std::flush;
      sortOrder(forest,splitOrder,forest->numLeaves);
      BARNEY_CUDA_SYNC_CHECK();

      std::cout << "splitting cells" << std::endl << std::flush;
      splitLeaves(forest,splitOrder);
      BARNEY_CUDA_CALL(Free(splitOrder));
      BARNEY_CUDA_SYNC_CHECK();
      // split nodes
      // resize
      // break;
    }

    std::cout << "final raster to fill all leaves..." << std::endl;
    clearLeaves(forest);
    BARNEY_CUDA_SYNC_CHECK();
    
    rasterMesh(forest,mesh);
    BARNEY_CUDA_SYNC_CHECK();
    
    std::cout << "dumping levels as boxes..." << std::endl;
    int maxLevel = 0;
    int minLevel = 1<<20;
    for (int i=0;i<forest->numLeaves;i++) {
      maxLevel = std::max(maxLevel,forest->leaves[i].level);
      minLevel = std::min(minLevel,forest->leaves[i].level);
    }
#if 0
    std::ofstream out("test_boxes",std::ios::binary);
    for (int iz=0;iz<4;iz++)
      for (int iy=0;iy<4;iy++)
        for (int ix=0;ix<4;ix++) {
          // PRINT(vec3f(ix,iy,iz));
          box3f box;
          box.lower = 2.f*vec3f(ix,iy,iz);
          box.upper = box.lower+vec3f(1.f);
          // PRINT(box);
          out.write((const char *)&box,sizeof(box));
        }
    out.close();
    exit(0);
#endif
    PRINT(forest->numLeaves);
    for (int level=minLevel;level<=maxLevel;level++) {
      std::cout << "dumping level " << level << std::endl;
      std::string outFileName = "adaptiveCells_level"+std::to_string(level);
      std::ofstream out(outFileName.c_str(),std::ios::binary);
      for (int i=0;i<forest->numLeaves;i++) {
        auto leaf = forest->leaves[i];
        box3f box = forest->getLeafBounds(leaf);
        // std::cout << "  " << leaf.cellID << ";" << leaf.level << "  " << box << std::endl;
        // if (leaf.duringBuild.numElements == 0) continue;

        // PRINT(leaf.cellID);
        if (leaf.level != level) continue;
        out.write((const char *)&box,sizeof(box));
      }
      out.close();
    }
    std::cout << "dumped all levels... exiting" << std::endl;
    exit(0);
  }
  
}



