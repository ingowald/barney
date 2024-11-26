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

#include "barney/umesh/os/AWT.h"

#define BUFFER_CREATE owlDeviceBufferCreate
// #define BUFFER_CREATE owlManagedMemoryBufferCreate

namespace barney {

  extern "C" char AWT_ptx[];

  void UMeshAWT::DD::addVars(std::vector<OWLVarDecl> &vars, int base)
  {
    Inherited::addVars(vars,base);
    vars.push_back({ "nodes", OWL_BUFPTR, base+OWL_OFFSETOF(DD,nodes) });
    vars.push_back({ "roots", OWL_BUFPTR, base+OWL_OFFSETOF(DD,roots) });
  }

  OWLGeomType UMeshAWT::Host::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'UMeshAWT' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;
    
    // static OWLVarDecl params[]
    //   = {
    //      { "mesh.worldBounds.lower", OWL_FLOAT4, OWL_OFFSETOF(DD,mesh.worldBounds.lower) },
    //      { "mesh.worldBounds.upper", OWL_FLOAT4, OWL_OFFSETOF(DD,mesh.worldBounds.upper) },
    //      { "mesh.vertices", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.vertices) },
    //      { "mesh.tetIndices", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.tetIndices) },
    //      { "mesh.pyrIndices", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.pyrIndices) },
    //      { "mesh.wedIndices", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.wedIndices) },
    //      { "mesh.hexIndices", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.hexIndices) },
    //      { "mesh.elements", OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.elements) },
    //      { "mesh.gridOffsets",    OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.gridOffsets) },
    //      { "mesh.gridDims",    OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.gridDims) },
    //      { "mesh.gridDomains",    OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.gridDomains) },
    //      { "mesh.gridScalars",    OWL_BUFPTR, OWL_OFFSETOF(DD,mesh.gridScalars) },
    //      { "mesh.numElements", OWL_INT, OWL_OFFSETOF(DD,mesh.numElements) },
    //      { "nodes", OWL_BUFPTR, OWL_OFFSETOF(DD,nodes) },
    //      { "roots", OWL_BUFPTR, OWL_OFFSETOF(DD,roots) },
    //      { "xf.values", OWL_BUFPTR, OWL_OFFSETOF(DD,xf.values) },
    //      { "xf.domain", OWL_FLOAT2, OWL_OFFSETOF(DD,xf.domain) },
    //      { "xf.baseDensity", OWL_FLOAT, OWL_OFFSETOF(DD,xf.baseDensity) },
    //      { "xf.numValues", OWL_INT, OWL_OFFSETOF(DD,xf.numValues) },
    //      { nullptr }
    // };
    std::vector<OWLVarDecl> params;
    UMeshAWT::DD::addVars(params,0);
    
    OWLModule module = owlModuleCreate
      (devGroup->owl,AWT_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(UMeshAWT::DD),
       params.data(),(int)params.size());
    owlGeomTypeSetBoundsProg(gt,module,"UMeshAWTBounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"UMeshAWTIsec");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"UMeshAWTCH");
    owlBuildPrograms(devGroup->owl);
    
    return gt;
  }

  struct BuildState : public AWTNode {
    int numUsed = 0;
  };
  
  void UMeshAWT::Host::buildNodes(cuBQL::WideBVH<float,3, 4> &qbvh)
  {
    nodes.resize(qbvh.numNodes);
    for (int nodeID=0;nodeID<(int)qbvh.numNodes;nodeID++)
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

  void UMeshAWT::Host::extractRoots()
  {
    int desiredRootDepth = AWT_DEFAULT_MAX_DEPTH;
    char *fromEnv = getenv("AWT_MAX_DEPTH");
    if (fromEnv)
      desiredRootDepth = std::stoi(fromEnv);
    PRINT(desiredRootDepth);
    
    
    std::vector<int> depthOf(nodes.size());
    std::fill(depthOf.begin(),depthOf.end(),-1);
    while (true) {
      volatile int changedOne = false;
      owl::parallel_for_blocked
        (0,int(nodes.size()),1024,
         [&](int begin, int end) {
           for (int nodeID=begin;nodeID<end;nodeID++) {
             auto &node = nodes[nodeID];
             int depth = 0;
             for (int childID=0;childID<4;childID++) {
               auto &child = node.child[childID];
               if (!child.valid()) continue;
               if (child.isLeaf()) continue;
               assert(child.offset < depthOf.size());
               depth = std::max(depth,1+depthOf[child.offset]);
             }
             if (depth != depthOf[nodeID]) {
               depthOf[nodeID] = depth;
               if (!changedOne) changedOne = true;
             }
           }
         });
      if (!changedOne) break;
    }
    // now have proper depth for each node

    for (int nodeID=0;nodeID<nodes.size();nodeID++) {
      auto &node = nodes[nodeID];
      for (int childID=0;childID<4;childID++) {
        auto &child = node.child[childID];
        if (!child.valid())
          // cetainly not a leaf ...
          continue;

        if ((child.isLeaf() || (depthOf[child.offset] < desiredRootDepth))
            &&
            ((nodeID == 0) || (depthOf[nodeID] >= desiredRootDepth)))
          roots.push_back((nodeID<<2) | childID);
      }
    }
    std::cout << "#bn.awt: number of roots found " << roots.size() << std::endl;
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
      if (!node.child[i].valid())//node.depth[i] < 0)
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
  
  void UMeshAWT::Host::buildAWT()
  {
    double t0 = getCurrentTime();
    
    SetActiveGPU forDuration(devGroup->devices[0]);
    // ==================================================================
    
    cuBQL::WideBVH<float,3,4> bvh;
    box3f *d_primBounds = 0;
    range1f *d_primRanges = 0;
    BARNEY_CUDA_CALL(MallocManaged(&d_primBounds,mesh->elements.size()*sizeof(box3f)));
    BARNEY_CUDA_CALL(MallocManaged(&d_primRanges,mesh->elements.size()*sizeof(range1f)));
    
    auto d_mesh = mesh->getDD(getDevices()[0]);
    mesh->computeElementBBs(0,d_primBounds,d_primRanges);
    
    cuBQL::BuildConfig buildConfig;
    buildConfig.makeLeafThreshold = AWTNode::max_leaf_size;
    // buildConfig.enableSAH();
#if BARNEY_CUBQL_HOST
    cuBQL::host::spatialMedian(bvh,
                               (const cuBQL::box_t<float,3>*)d_primBounds,
                               (uint32_t)mesh->elements.size(),
                               buildConfig);
#else
    static cuBQL::ManagedMemMemoryResource managedMem;
    std::cout << "#bn.awt: going to build cuBQL BVH over "
              << prettyNumber(mesh->elements.size()) << " elements" << std::endl;
    cuBQL::gpuBuilder(bvh,
                      (const cuBQL::box_t<float,3>*)d_primBounds,
                      (uint32_t)mesh->elements.size(),
                      buildConfig,
                      (cudaStream_t)0,
                      managedMem);
#endif
    std::cout << "#bn.awt: building (host-)AWT nodes from cubql bvh (on the host rn :/)" << std::endl;
    buildNodes(bvh);
    std::cout << "#bn.awt: extracting awt roots (on the host rn :/)" << std::endl;
    extractRoots();
    std::cout << "#bn.awt: refitting awt ranges (on the host rn :-/)" << std::endl;
    refitRanges(nodes,bvh.primIDs,d_primBounds,d_primRanges);

    std::cout << "#bn.awt: re-ordering elements" << std::endl;
    std::vector<Element> reorderedElements(mesh->elements.size());
    for (int i=0;i<mesh->elements.size();i++) {
      reorderedElements[i] = mesh->elements[bvh.primIDs[i]];
    }
    mesh->elements = reorderedElements;
    std::cout << "#bn.awt: uploading reordered element refs to OWL" << std::endl;
    owlBufferUpload(mesh->elementsBuffer,reorderedElements.data());
    BARNEY_CUDA_CALL(Free(d_primBounds));
    BARNEY_CUDA_CALL(Free(d_primRanges));

    
#if BARNEY_CUBQL_HOST
    cuBQL::host::freeBVH(bvh);
#else
    cuBQL::free(bvh,0,managedMem);
#endif
    
    // ==================================================================

    assert(sizeof(roots[0]) == sizeof(int));
    rootsBuffer = owlDeviceBufferCreate(devGroup->owl,OWL_INT,
                                        roots.size(),roots.data());
    std::cout << "#bn.awt: uploading AWT nodes buffer to OWL" << std::endl;
    nodesBuffer = BUFFER_CREATE(devGroup->owl,OWL_USER_TYPE(AWTNode),
                                nodes.size(),nodes.data());
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
    if (!node.child[cID].valid())//depth[cID] < 0) 
      node.majorant[cID] = 0.f;
    else
      node.majorant[cID] = xf.majorant(getRange(node.bounds[cID]));
  }

  void UMeshAWT::Host::setVariables(OWLGeom geom)
  {
    Inherited::setVariables(geom);
    owlGeomSetBuffer(geom,"roots",rootsBuffer);
    owlGeomSetBuffer(geom,"nodes",nodesBuffer);
  }
  
  void UMeshAWT::Host::build(bool full_rebuild)
  {
    if (!full_rebuild) return;
    
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
      setVariables(geom);

      // // ------------------------------------------------------------------
      // assert(mesh->tetIndicesBuffer);
      // owlGeomSet4fv(geom,"mesh.worldBounds.lower",&mesh->worldBounds.lower.x);
      // owlGeomSet4fv(geom,"mesh.worldBounds.upper",&mesh->worldBounds.upper.x);
      // owlGeomSetBuffer(geom,"mesh.vertices",mesh->verticesBuffer);
      
      // owlGeomSetBuffer(geom,"mesh.tetIndices",mesh->tetIndicesBuffer);
      // owlGeomSetBuffer(geom,"mesh.pyrIndices",mesh->pyrIndicesBuffer);
      // owlGeomSetBuffer(geom,"mesh.wedIndices",mesh->wedIndicesBuffer);
      // owlGeomSetBuffer(geom,"mesh.hexIndices",mesh->hexIndicesBuffer);
      // owlGeomSetBuffer(geom,"mesh.elements",mesh->elementsBuffer);
      // owlGeomSetBuffer(geom,"mesh.gridOffsets",mesh->gridOffsetsBuffer);
      // owlGeomSetBuffer(geom,"mesh.gridDims",mesh->gridDimsBuffer);
      // owlGeomSetBuffer(geom,"mesh.gridDomains",mesh->gridDomainsBuffer);
      // owlGeomSetBuffer(geom,"mesh.gridScalars",mesh->gridScalarsBuffer);
      // // ------------------------------------------------------------------      
      
      // ------------------------------------------------------------------      
      
      // if (volume->xf.domain.lower < volume->xf.domain.upper) {
      //   owlGeomSet2f(geom,"xf.domain",volume->xf.domain.lower,volume->xf.domain.upper);
      // } else {
      //   owlGeomSet2f(geom,"xf.domain",mesh->worldBounds.lower.w,mesh->worldBounds.upper.w);
      // }
      // owlGeomSet1f(geom,"xf.baseDensity",volume->xf.baseDensity);
      // owlGeomSet1i(geom,"xf.numValues",(int)volume->xf.values.size());
      // // intentionally set to null for first-time build
      // owlGeomSetBuffer(geom,"xf.values",0/*volume->xf.valuesBuffer*/);
      
      // ------------------------------------------------------------------      
      group
        = owlUserGeomGroupCreate(devGroup->owl,1,&geom,OPTIX_BUILD_FLAG_ALLOW_UPDATE);
      owlGroupBuildAccel(group);
      volume->generatedGroups = { group }; 
    }
    else
      std::cout << "original awt build already done - just need refit" << std::endl;

    setVariables(geom);

    // if (volume->xf.domain.lower < volume->xf.domain.upper) {
    //   owlGeomSet2f(geom,"xf.domain",volume->xf.domain.lower,volume->xf.domain.upper);
    // } else {
    //   owlGeomSet2f(geom,"xf.domain",mesh->worldBounds.lower.w,mesh->worldBounds.upper.w);
    // }
    // owlGeomSet1f(geom,"xf.baseDensity",volume->xf.baseDensity);
    // owlGeomSet1i(geom,"xf.numValues",(int)volume->xf.values.size());
    // owlGeomSetBuffer(geom,"xf.values",volume->xf.valuesBuffer);

    
    for (auto dev : getDevices()) {
      SetActiveGPU forDuration(dev);
      CHECK_CUDA_LAUNCH(recomputeMajorants,
                        divRoundUp(int(4*nodes.size()),1024),1024,0,0,
                        (AWTNode*)owlBufferGetPointer(nodesBuffer,dev->owlID),
         (int)nodes.size(),
         volume->xf.getDD(dev));
      // recomputeMajorants<<<divRoundUp(int(4*nodes.size()),1024),1024>>>
      //   ((AWTNode*)owlBufferGetPointer(nodesBuffer,dev->owlID),
      //    (int)nodes.size(),
      //    volume->xf.getDD(dev));
    }
    std::cout << "refitting ... umesh awt/object space geom" << std::endl;
    owlGroupRefitAccel(volume->generatedGroups[0]);
  }

}
