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

#include "barney/umesh/RTXObjectSpace.h"

namespace barney {

  extern "C" char RTXObjectSpace_ptx[];
  
  OWLGeomType RTXObjectSpace::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'RTXObjectSpace' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;
    
    static OWLVarDecl params[]
      = {
         { "mesh.worldBounds.lower", OWL_FLOAT4, OWL_OFFSETOF(DD,mesh.worldBounds.lower) },
         { "mesh.worldBounds.upper", OWL_FLOAT4, OWL_OFFSETOF(DD,mesh.worldBounds.upper) },
         { "mesh.vertices", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.vertices) },
         { "mesh.tetIndices", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.tetIndices) },
         { "mesh.pyrIndices", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.pyrIndices) },
         { "mesh.wedIndices", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.wedIndices) },
         { "mesh.hexIndices", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.hexIndices) },
         { "mesh.elements", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.elements) },
         { "mesh.gridOffsets",    OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.gridOffsets) },
         { "mesh.gridDims",    OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.gridDims) },
         { "mesh.gridDomains",    OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.gridDomains) },
         { "mesh.gridScalars",    OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.gridScalars) },
         { "mesh.numElements", OWL_INT, OWL_OFFSETOF(DD,mesh.numElements) },
         { "clusters", OWL_BUFPTR, OWL_OFFSETOF(DD,clusters) },
         { "xf.values", OWL_BUFPTR, OWL_OFFSETOF(DD,xf.values) },
         { "xf.domain", OWL_FLOAT2, OWL_OFFSETOF(DD,xf.domain) },
         { "xf.baseDensity", OWL_FLOAT, OWL_OFFSETOF(DD,xf.baseDensity) },
         { "xf.numValues", OWL_INT, OWL_OFFSETOF(DD,xf.numValues) },
         { nullptr }
    };
    OWLModule module = owlModuleCreate
      (devGroup->owl,RTXObjectSpace_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(RTXObjectSpace::DD),
       params,-1);
    owlGeomTypeSetBoundsProg(gt,module,"RTXObjectSpaceBounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"RTXObjectSpaceIsec");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"RTXObjectSpaceCH");
    owlBuildPrograms(devGroup->owl);
    
    return gt;
  }

  OWLGeomType UMeshAWT::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'UMeshAWT' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;
    
    static OWLVarDecl params[]
      = {
         { "mesh.worldBounds.lower", OWL_FLOAT4, OWL_OFFSETOF(DD,mesh.worldBounds.lower) },
         { "mesh.worldBounds.upper", OWL_FLOAT4, OWL_OFFSETOF(DD,mesh.worldBounds.upper) },
         { "mesh.vertices", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.vertices) },
         { "mesh.tetIndices", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.tetIndices) },
         { "mesh.pyrIndices", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.pyrIndices) },
         { "mesh.wedIndices", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.wedIndices) },
         { "mesh.hexIndices", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.hexIndices) },
         { "mesh.elements", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.elements) },
         { "mesh.gridOffsets",    OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.gridOffsets) },
         { "mesh.gridDims",    OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.gridDims) },
         { "mesh.gridDomains",    OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.gridDomains) },
         { "mesh.gridScalars",    OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.gridScalars) },
         { "mesh.numElements", OWL_INT, OWL_OFFSETOF(DD,mesh.numElements) },
         { "nodes", OWL_BUFPTR, OWL_OFFSETOF(DD,nodes) },
         { "roots", OWL_BUFPTR, OWL_OFFSETOF(DD,roots) },
         { "xf.values", OWL_BUFPTR, OWL_OFFSETOF(DD,xf.values) },
         { "xf.domain", OWL_FLOAT2, OWL_OFFSETOF(DD,xf.domain) },
         { "xf.baseDensity", OWL_FLOAT, OWL_OFFSETOF(DD,xf.baseDensity) },
         { "xf.numValues", OWL_INT, OWL_OFFSETOF(DD,xf.numValues) },
         { nullptr }
    };
    OWLModule module = owlModuleCreate
      (devGroup->owl,RTXObjectSpace_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(UMeshAWT::DD),
       params,-1);
    owlGeomTypeSetBoundsProg(gt,module,"UMeshAWTBounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"UMeshAWTIsec");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"UMeshAWTCH");
    owlBuildPrograms(devGroup->owl);
    
    return gt;
  }

  inline box3f make_box(cuBQL::box3f b)
  {
    return (const box3f&)b;
  }
  inline box4f make_box4f(box3f b, range1f r=range1f())
  {
    box4f b4;
    (vec3f&)b4.lower = (const vec3f&)b.lower;
    (vec3f&)b4.upper = (const vec3f&)b.upper;
    b4.lower.w = r.lower;
    b4.upper.w = r.upper;
    return b4;
  }
  
  struct BuildState : public AWTNode {
    int numUsed = 0;
  };
  
  void UMeshAWT::buildNodes(cuBQL::WideBVH<float,3, 4> &qbvh)
  {
    // PING;
    // PRINT(qbvh.numNodes);
    nodes.resize(qbvh.numNodes);
    for (int nodeID=0;nodeID<qbvh.numNodes;nodeID++)
      for (int childID=0;childID<4;childID++) {
        box3f bounds = make_box(qbvh.nodes[nodeID].children[childID].bounds);
        if (!qbvh.nodes[nodeID].children[childID].valid)
          bounds = box3f();
        
        nodes[nodeID].bounds[childID] = make_box4f(bounds);
        if (bounds.empty()) {
          nodes[nodeID].child[childID].offset = 0;
          nodes[nodeID].child[childID].count  = 0;
        } else {
          nodes[nodeID].child[childID].offset = qbvh.nodes[nodeID].children[childID].offset;
          nodes[nodeID].child[childID].count = qbvh.nodes[nodeID].children[childID].count;
        }
      }
  }

  int UMeshAWT::extractRoots(cuBQL::WideBVH<float,3, 4> &qbvh,
                                        int nodeID)
  {
    // PING; PRINT(nodeID);
    auto &node = nodes[nodeID];
    int maxDepth = 0;
    for (int i=0;i<4;i++) {
      // PRINT(i);
      // PRINT(node.bounds[i]);
      // PRINT(node.child[i].count);
      if (getBox(node.bounds[i]).empty()) {
        node.depth[i] = -1;
      } else if (node.child[i].count > 0) {
        node.depth[i] = 0;
      } else {
        node.depth[i] = extractRoots(qbvh,node.child[i].offset);
      }
      maxDepth = std::max(maxDepth,node.depth[i]);
    }
    if (maxDepth < AWT_MAX_DEPTH && nodeID != 0)
      // can still merge on parent
      return maxDepth+1;
    
    for (int i=0;i<4;i++) {
      if (node.depth[i] == -1)
        // certainly not a root....
        continue;

      if (node.depth[i] <= AWT_MAX_DEPTH)
        roots.push_back((nodeID<<2)|i);
    }
    return maxDepth+1;
  }

  box4f refitRanges(std::vector<AWTNode> &nodes,
                    uint32_t *primIDs,
                    box3f *d_primBounds,
                    range1f *d_primRanges,
                    int nodeID=0)
  {
    box4f nb;
    AWTNode &node = nodes[nodeID];
    for (int i=0;i<4;i++) {
      if (node.depth[i] < 0)
        continue;
      
      int ofs = node.child[i].offset;
      int cnt = node.child[i].count;
      if (cnt == 0) {
        node.bounds[i]
          = refitRanges(nodes,primIDs,d_primBounds,d_primRanges,
                        ofs);
      } else {
        box4f leafBounds;
        for (int j=0;j<cnt;j++) {
          int pid = primIDs[ofs+j];
          leafBounds.extend(make_box4f(d_primBounds[pid],
                                       d_primRanges[pid]));
        }
        node.bounds[i] = leafBounds;
      }
      nb.extend(node.bounds[i]);
    }
    return nb;
  }
  
  void UMeshAWT::buildAWT()
  {
    double t0 = getCurrentTime();
    
    SetActiveGPU forDuration(devGroup->devices[0]);
    // ==================================================================
    
    // buildHierarchy(nodes,roots,clusters,bvh);
    cuBQL::WideBVH<float,3,4> bvh;
    box3f *d_primBounds = 0;
    range1f *d_primRanges = 0;
    PING; PRINT(prettyDouble(getCurrentTime()-t0));
    BARNEY_CUDA_CALL(MallocManaged(&d_primBounds,mesh->elements.size()*sizeof(box3f)));
    BARNEY_CUDA_CALL(MallocManaged(&d_primRanges,mesh->elements.size()*sizeof(range1f)));
    
    auto d_mesh = mesh->getDD(0);
    mesh->computeElementBBs(0,d_primBounds,d_primRanges);
      // <<<divRoundUp((int)mesh->elements.size(),1024),1024>>>
      // (d_primBounds,d_primRanges,d_mesh);
    
    PING; PRINT(prettyDouble(getCurrentTime()-t0));
    cuBQL::BuildConfig buildConfig;
    buildConfig.makeLeafThreshold = AWTNode::max_leaf_size;
    // buildConfig.enableSAH();
    static cuBQL::ManagedMemMemoryResource managedMem;
    cuBQL::gpuBuilder(bvh,
                      (const cuBQL::box_t<float,3>*)d_primBounds,
                      (uint32_t)mesh->elements.size(),
                      buildConfig,
                      (cudaStream_t)0,
                      managedMem);

    PING; PRINT(prettyDouble(getCurrentTime()-t0));
    buildNodes(bvh);
    PING; PRINT(prettyDouble(getCurrentTime()-t0));
    extractRoots(bvh,0);
    PING; PRINT(prettyDouble(getCurrentTime()-t0));
    refitRanges(nodes,bvh.primIDs,d_primBounds,d_primRanges);
    PING; PRINT(prettyDouble(getCurrentTime()-t0));

    std::vector<Element> reorderedElements(mesh->elements.size());
    for (int i=0;i<mesh->elements.size();i++) {
      reorderedElements[i] = mesh->elements[bvh.primIDs[i]];
    }
    PING; PRINT(prettyDouble(getCurrentTime()-t0));
    mesh->elements = reorderedElements;
    owlBufferUpload(mesh->elementsBuffer,reorderedElements.data());
    BARNEY_CUDA_CALL(Free(d_primBounds));
    BARNEY_CUDA_CALL(Free(d_primRanges));

    
    cuBQL::free(bvh,0,managedMem);
    
    // ==================================================================

    assert(sizeof(roots[0]) == sizeof(int));
    rootsBuffer = owlDeviceBufferCreate(devGroup->owl,OWL_INT,
                                           roots.size(),roots.data());
    nodesBuffer = owlDeviceBufferCreate(devGroup->owl,OWL_USER_TYPE(AWTNode),
                                        nodes.size(),nodes.data());
    PING; PRINT(prettyDouble(getCurrentTime()-t0));
  }

  void RTXObjectSpace::createClusters()
  {
    assert(clusters.empty());
    assert(!clustersBuffer);

    SetActiveGPU forDuration(devGroup->devices[0]);
    // ==================================================================
    
    cuBQL::BinaryBVH<float,3> bvh;
    box3f *d_primBounds = 0;
    PING;
    BARNEY_CUDA_CALL(MallocManaged(&d_primBounds,mesh->elements.size()*sizeof(box3f)));
    
    auto d_mesh = mesh->getDD(0);
    mesh->computeElementBBs(0,d_primBounds);
    // computeElementBoundingBoxes
    //   <<<divRoundUp((int)mesh->elements.size(),1024),1024>>>
    //   (d_primBounds,d_mesh);
    
    cuBQL::BuildConfig buildConfig;
    buildConfig.makeLeafThreshold = 8;
    buildConfig.enableSAH();
    static cuBQL::ManagedMemMemoryResource managedMem;
    cuBQL::gpuBuilder(bvh,
                      (const cuBQL::box_t<float,3>*)d_primBounds,
                      (uint32_t)mesh->elements.size(),
                      buildConfig,
                      (cudaStream_t)0,
                      managedMem);
    std::vector<Element> reorderedElements(mesh->elements.size());
    for (int i=0;i<mesh->elements.size();i++) {
      reorderedElements[i] = mesh->elements[bvh.primIDs[i]];
    }
    mesh->elements = reorderedElements;
    owlBufferUpload(mesh->elementsBuffer,reorderedElements.data());
    BARNEY_CUDA_CALL(Free(d_primBounds));

    for (int i=0;i<bvh.numNodes;i++) {
      auto node = bvh.nodes[i];
      if (node.count == 0) continue;
      Cluster c;
      c.begin = node.offset;
      c.end = node.offset + node.count;
      clusters.push_back(c);
    }
    cuBQL::free(bvh,0,managedMem);
    
    // ==================================================================

    clustersBuffer = owlDeviceBufferCreate(devGroup->owl,OWL_USER_TYPE(Cluster),
                                           clusters.size(),clusters.data());
  }


  __global__
  void recomputeMajorants(AWTNode *nodes,
                          int numNodes,
                          TransferFunction::DD xf)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    int nodeID = tid / 4;
    int cID = tid % 4;
    if (nodeID >= numNodes)
      return;
    auto &node = nodes[nodeID];
    if (node.depth[cID] < 0) 
      node.majorant[cID] = 0.f;
    else
      node.majorant[cID] = xf.majorant(getRange(node.bounds[cID]));
  }

  void UMeshAWT::build()
  {
    BARNEY_CUDA_SYNC_CHECK();
    
    if (!group) {
      buildAWT();
      
      std::string gtTypeName = "UMeshAWT";
      OWLGeomType gt = devGroup->getOrCreateGeomTypeFor
        (gtTypeName,createGeomType);
      geom
        = owlGeomCreate(devGroup->owl,gt);
      int numPrims = (int)roots.size();
      owlGeomSetPrimCount(geom,numPrims);

      // ------------------------------------------------------------------
      assert(mesh->tetIndicesBuffer);
      owlGeomSet4fv(geom,"mesh.worldBounds.lower",&mesh->worldBounds.lower.x);
      owlGeomSet4fv(geom,"mesh.worldBounds.upper",&mesh->worldBounds.upper.x);
      owlGeomSetBuffer(geom,"mesh.vertices",mesh->verticesBuffer);
      
      owlGeomSetBuffer(geom,"mesh.tetIndices",mesh->tetIndicesBuffer);
      owlGeomSetBuffer(geom,"mesh.pyrIndices",mesh->pyrIndicesBuffer);
      owlGeomSetBuffer(geom,"mesh.wedIndices",mesh->wedIndicesBuffer);
      owlGeomSetBuffer(geom,"mesh.hexIndices",mesh->hexIndicesBuffer);
      owlGeomSetBuffer(geom,"mesh.elements",mesh->elementsBuffer);
      owlGeomSetBuffer(geom,"mesh.gridOffsets",mesh->gridOffsetsBuffer);
      owlGeomSetBuffer(geom,"mesh.gridDims",mesh->gridDimsBuffer);
      owlGeomSetBuffer(geom,"mesh.gridDomains",mesh->gridDomainsBuffer);
      owlGeomSetBuffer(geom,"mesh.gridScalars",mesh->gridScalarsBuffer);
      // ------------------------------------------------------------------      
      owlGeomSetBuffer(geom,"roots",rootsBuffer);
      owlGeomSetBuffer(geom,"nodes",nodesBuffer);
      
      // ------------------------------------------------------------------      
      
      if (volume->xf.domain.lower < volume->xf.domain.upper) {
        owlGeomSet2f(geom,"xf.domain",volume->xf.domain.lower,volume->xf.domain.upper);
      } else {
        owlGeomSet2f(geom,"xf.domain",mesh->worldBounds.lower.w,mesh->worldBounds.upper.w);
      }
      owlGeomSet1f(geom,"xf.baseDensity",volume->xf.baseDensity);
      owlGeomSet1i(geom,"xf.numValues",(int)volume->xf.values.size());
      // intentionally set to null for first-time build
      owlGeomSetBuffer(geom,"xf.values",0/*volume->xf.valuesBuffer*/);
      
      // ------------------------------------------------------------------      
      group
        = owlUserGeomGroupCreate(devGroup->owl,1,&geom,OPTIX_BUILD_FLAG_ALLOW_UPDATE);
      owlGroupBuildAccel(group);
      volume->generatedGroups.push_back(group);
    }
    
    if (volume->xf.domain.lower < volume->xf.domain.upper) {
      owlGeomSet2f(geom,"xf.domain",volume->xf.domain.lower,volume->xf.domain.upper);
    } else {
      owlGeomSet2f(geom,"xf.domain",mesh->worldBounds.lower.w,mesh->worldBounds.upper.w);
    }
    owlGeomSet1f(geom,"xf.baseDensity",volume->xf.baseDensity);
    owlGeomSet1i(geom,"xf.numValues",(int)volume->xf.values.size());
    owlGeomSetBuffer(geom,"xf.values",volume->xf.valuesBuffer);

    std::cout << "RECOMPUTING AWT MAJORANTS!\n" << std::endl;
    PRINT(nodes.size());
    PRINT(roots.size());
    for (int devID = 0;devID<devGroup->devices.size(); devID++) {
      auto dev = devGroup->devices[devID];
      SetActiveGPU forDuration(dev);
      recomputeMajorants<<<divRoundUp(int(4*nodes.size()),1024),1024>>>
        ((AWTNode*)owlBufferGetPointer(nodesBuffer,devID),
         nodes.size(),
         volume->xf.getDD(devID));
    }
    std::cout << "refitting ... umesh awt/object space geom" << std::endl;
    owlGroupRefitAccel(volume->generatedGroups[0]);
  }





  void RTXObjectSpace::build()
  {
    BARNEY_CUDA_SYNC_CHECK();
    
    if (!group) {
      createClusters();
      
      std::string gtTypeName = "RTXObjectSpace";
      OWLGeomType gt = devGroup->getOrCreateGeomTypeFor
        (gtTypeName,createGeomType);
      geom
        = owlGeomCreate(devGroup->owl,gt);
      int numPrims = (int)clusters.size();
      PRINT(numPrims);
      owlGeomSetPrimCount(geom,numPrims);

      // ------------------------------------------------------------------
      assert(mesh->tetIndicesBuffer);
      owlGeomSet4fv(geom,"mesh.worldBounds.lower",&mesh->worldBounds.lower.x);
      owlGeomSet4fv(geom,"mesh.worldBounds.upper",&mesh->worldBounds.upper.x);
      owlGeomSetBuffer(geom,"mesh.vertices",mesh->verticesBuffer);
      
      owlGeomSetBuffer(geom,"mesh.tetIndices",mesh->tetIndicesBuffer);
      owlGeomSetBuffer(geom,"mesh.pyrIndices",mesh->pyrIndicesBuffer);
      owlGeomSetBuffer(geom,"mesh.wedIndices",mesh->wedIndicesBuffer);
      owlGeomSetBuffer(geom,"mesh.hexIndices",mesh->hexIndicesBuffer);
      owlGeomSetBuffer(geom,"mesh.elements",mesh->elementsBuffer);
      owlGeomSetBuffer(geom,"mesh.gridOffsets",mesh->gridOffsetsBuffer);
      owlGeomSetBuffer(geom,"mesh.gridDims",mesh->gridDimsBuffer);
      owlGeomSetBuffer(geom,"mesh.gridDomains",mesh->gridDomainsBuffer);
      owlGeomSetBuffer(geom,"mesh.gridScalars",mesh->gridScalarsBuffer);
      // ------------------------------------------------------------------      
      owlGeomSetBuffer(geom,"clusters",clustersBuffer);
      
      // ------------------------------------------------------------------      
      
      if (volume->xf.domain.lower < volume->xf.domain.upper) {
        owlGeomSet2f(geom,"xf.domain",volume->xf.domain.lower,volume->xf.domain.upper);
      } else {
        owlGeomSet2f(geom,"xf.domain",mesh->worldBounds.lower.w,mesh->worldBounds.upper.w);
      }
      owlGeomSet1f(geom,"xf.baseDensity",volume->xf.baseDensity);
      owlGeomSet1i(geom,"xf.numValues",(int)volume->xf.values.size());
      // intentionally set to null for first-time build
      owlGeomSetBuffer(geom,"xf.values",0/*volume->xf.valuesBuffer*/);
      
      // ------------------------------------------------------------------      
      group
        = owlUserGeomGroupCreate(devGroup->owl,1,&geom,OPTIX_BUILD_FLAG_ALLOW_UPDATE);
      owlGroupBuildAccel(group);
      volume->generatedGroups.push_back(group);
    }
    
    if (volume->xf.domain.lower < volume->xf.domain.upper) {
      owlGeomSet2f(geom,"xf.domain",volume->xf.domain.lower,volume->xf.domain.upper);
    } else {
      owlGeomSet2f(geom,"xf.domain",mesh->worldBounds.lower.w,mesh->worldBounds.upper.w);
    }
    owlGeomSet1f(geom,"xf.baseDensity",volume->xf.baseDensity);
    owlGeomSet1i(geom,"xf.numValues",(int)volume->xf.values.size());
    owlGeomSetBuffer(geom,"xf.values",volume->xf.valuesBuffer);

    std::cout << "refitting ... umesh object space geom" << std::endl;
    // owlGroupBuildAccel(group);
    owlGroupRefitAccel(group);
  }

      
}

