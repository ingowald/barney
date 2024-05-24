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

#include "barney/umesh/os/RTXObjectSpace.h"

namespace barney {
    
  extern "C" char RTXObjectSpace_ptx[];
  
  void RTXObjectSpace::DD::addVars(std::vector<OWLVarDecl> &vars, int base)
  {
    Inherited::addVars(vars,base);
    vars.push_back
      ({ "clusters",       OWL_BUFPTR, base+OWL_OFFSETOF(DD,clusters) });
    vars.push_back
      ({ "firstTimeBuild", OWL_INT,    base+OWL_OFFSETOF(DD,firstTimeBuild) });
  }

  OWLGeomType RTXObjectSpace::Host::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'RTXObjectSpace' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;

    std::vector<OWLVarDecl> params;
    RTXObjectSpace::DD::addVars(params,0);
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
    //      { "clusters", OWL_BUFPTR, OWL_OFFSETOF(DD,clusters) },
    //      { "xf.values", OWL_BUFPTR, OWL_OFFSETOF(DD,xf.values) },
    //      { "xf.domain", OWL_FLOAT2, OWL_OFFSETOF(DD,xf.domain) },
    //      { "xf.baseDensity", OWL_FLOAT, OWL_OFFSETOF(DD,xf.baseDensity) },
    //      { "xf.numValues", OWL_INT, OWL_OFFSETOF(DD,xf.numValues) },
    //      { nullptr }
    // };
    OWLModule module = owlModuleCreate
      (devGroup->owl,RTXObjectSpace_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(RTXObjectSpace::DD),
       params.data(),(int)params.size());
    owlGeomTypeSetBoundsProg(gt,module,"RTXObjectSpaceBounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"RTXObjectSpaceIsec");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"RTXObjectSpaceCH");
    owlBuildPrograms(devGroup->owl);
    
    return gt;
  }

  void RTXObjectSpace::Host::createClusters()
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

    for (int i=0;i<(int)bvh.numNodes;i++) {
      auto node = bvh.nodes[i];
      if (node.count == 0) continue;
      Cluster c;
      c.begin = int(node.offset);
      c.end = int(node.offset + node.count);
      clusters.push_back(c);
    }
    cuBQL::free(bvh,0,managedMem);
    
    // ==================================================================

    clustersBuffer = owlDeviceBufferCreate(devGroup->owl,OWL_USER_TYPE(Cluster),
                                           clusters.size(),clusters.data());
    PING; PRINT(clustersBuffer);
  }

  void RTXObjectSpace::Host::setVariables(OWLGeom geom)
  {
    Inherited::setVariables(geom);
    owlGeomSetBuffer(geom,"clusters",clustersBuffer);
    owlGeomSet1i(geom,"firstTimeBuild",(int)firstTimeBuild);
  }
  

  void RTXObjectSpace::Host::build(bool full_rebuild)
  {
    if (!full_rebuild) return;
    
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
      
      firstTimeBuild = true;
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
      // owlGeomSetBuffer(geom,"clusters",clustersBuffer);
      
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

      firstTimeBuild = false;
    }

    volume->generatedGroups = { group };
    setVariables(geom);
    
    // if (volume->xf.domain.lower < volume->xf.domain.upper) {
    //   owlGeomSet2f(geom,"xf.domain",volume->xf.domain.lower,volume->xf.domain.upper);
    // } else {
    //   owlGeomSet2f(geom,"xf.domain",mesh->worldBounds.lower.w,mesh->worldBounds.upper.w);
    // }
    // owlGeomSet1f(geom,"xf.baseDensity",volume->xf.baseDensity);
    // owlGeomSet1i(geom,"xf.numValues",(int)volume->xf.values.size());
    // owlGeomSetBuffer(geom,"xf.values",volume->xf.valuesBuffer);

    std::cout << "refitting ... umesh object space geom" << std::endl;
    // owlGroupBuildAccel(group);
    owlGroupRefitAccel(group);
  }

}

