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

#include "barney/unstructured/UMeshMCAccelerator.h"

namespace barney {
  
  extern "C" char UMeshMCAccelerator_ptx[];
  
  template<typename VolumeSampler>
  void UMeshMCAccelerator<VolumeSampler>::buildMCs() 
  {
    std::cout << OWL_TERMINAL_BLUE
              << "#bn.um: (re-)building macro cells\n"
              << OWL_TERMINAL_DEFAULT << std::endl;
    assert(mesh);
    assert(volume);
    mesh->buildInitialMacroCells(mcGrid);
    mcGrid.computeMajorants(&volume->xf);
  }

  template<typename VolumeSampler>
  OWLGeomType UMeshMCAccelerator<VolumeSampler>::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'UMesh_MC_CUBQL' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;

    // TODO - this needs to get split into different classes'
    // 'addParams()', so mesh, sampler, tec can all declare and set
    // set theit own
    static OWLVarDecl params[]
      = {
         { "mesh.vertices",    OWL_BUFPTR, OWL_OFFSETOF(DD,sampler.mesh.vertices) },
         { "mesh.tetIndices",  OWL_BUFPTR, OWL_OFFSETOF(DD,sampler.mesh.tetIndices) },
         { "mesh.hexIndices",  OWL_BUFPTR, OWL_OFFSETOF(DD,sampler.mesh.hexIndices) },
         { "mesh.elements",    OWL_BUFPTR, OWL_OFFSETOF(DD,sampler.mesh.elements) },
         { "mesh.worldBounds.lower", OWL_FLOAT4, OWL_OFFSETOF(DD,sampler.mesh.worldBounds.lower) },
         { "mesh.worldBounds.upper", OWL_FLOAT4, OWL_OFFSETOF(DD,sampler.mesh.worldBounds.upper) },
         { "numElements", OWL_INT, OWL_OFFSETOF(DD,sampler.mesh.numElements) },
         { "xf.values",   OWL_BUFPTR, OWL_OFFSETOF(DD,volume.xf.values) },
         { "xf.domain",   OWL_FLOAT2, OWL_OFFSETOF(DD,volume.xf.domain) },
         { "xf.baseDensity", OWL_FLOAT, OWL_OFFSETOF(DD,volume.xf.baseDensity) },
         { "xf.numValues", OWL_INT, OWL_OFFSETOF(DD,volume.xf.numValues) },
         { "bvhNodes",    OWL_BUFPTR, OWL_OFFSETOF(DD,sampler.bvhNodes) },
         { "mcGrid.dims", OWL_INT3, OWL_OFFSETOF(DD,mcGrid.dims) },
         { "mcGrid.majorants", OWL_BUFPTR, OWL_OFFSETOF(DD,mcGrid.majorants) },
         { nullptr }
    };
    OWLModule module = owlModuleCreate
      (devGroup->owl,UMeshMCAccelerator_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(DD),
       params,-1);
    owlGeomTypeSetBoundsProg(gt,module,"UMesh_MC_CUBQL_Bounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"UMesh_MC_CUBQL_Isec");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"UMesh_MC_CUBQL_CH");
    owlBuildPrograms(devGroup->owl);
    
    return gt;
  }
  
  template<typename VolumeSampler>
  void UMeshMCAccelerator<VolumeSampler>::build()
  {
    auto devGroup = mesh->devGroup;

    BARNEY_CUDA_SYNC_CHECK();
    buildMCs();
    BARNEY_CUDA_SYNC_CHECK();
    this->sampler.build();
    BARNEY_CUDA_SYNC_CHECK();
    
    if (volume->generatedGroups.empty()) {
      std::string gtTypeName = "UMesh_MC_CUBQL";
      OWLGeomType gt = devGroup->getOrCreateGeomTypeFor
        (gtTypeName,createGeomType);
      geom
        = owlGeomCreate(devGroup->owl,gt);
#if UMESH_MC_USE_DDA
      owlGeomSetPrimCount(geom,1);
#else
      vec3i dims = mcGrid.dims;
      int primCount = dims.x*dims.y*dims.z;
      PING; PRINT(dims); PRINT(primCount);
      owlGeomSetPrimCount(geom,primCount);
#endif

      // ------------------------------------------------------------------      
      owlGeomSet3iv(geom,"mcGrid.dims",&mcGrid.dims.x);
      // intentionally set to null for first-time build
      owlGeomSetBuffer(geom,"mcGrid.majorants",0);
      
      // ------------------------------------------------------------------
      assert(mesh->tetIndicesBuffer);
      owlGeomSet4fv(geom,"mesh.worldBounds.lower",&mesh->worldBounds.lower.x);
      owlGeomSet4fv(geom,"mesh.worldBounds.upper",&mesh->worldBounds.upper.x);
      owlGeomSetBuffer(geom,"mesh.vertices",mesh->verticesBuffer);
      
      owlGeomSetBuffer(geom,"mesh.tetIndices",mesh->tetIndicesBuffer);
      owlGeomSetBuffer(geom,"mesh.hexIndices",mesh->hexIndicesBuffer);
      owlGeomSetBuffer(geom,"mesh.elements",mesh->elementsBuffer);
      
      // ------------------------------------------------------------------      
      owlGeomSetBuffer(geom,"bvhNodes",this->sampler.bvhNodesBuffer);
      
      // ------------------------------------------------------------------      
      
      if (volume->xf.domain.lower < volume->xf.domain.upper) {
        owlGeomSet2f(geom,"xf.domain",volume->xf.domain.lower,volume->xf.domain.upper);
      } else {
        owlGeomSet2f(geom,"xf.domain",mesh->worldBounds.lower.w,mesh->worldBounds.upper.w);
      }
      owlGeomSet1f(geom,"xf.baseDensity",volume->xf.baseDensity);
      owlGeomSet1i(geom,"xf.numValues",(int)volume->xf.values.size());
      PING; PRINT(volume->xf.values.size());
      owlGeomSetBuffer(geom,"xf.values",volume->xf.valuesBuffer);
      
      // ------------------------------------------------------------------      
      OWLGroup group
        = owlUserGeomGroupCreate(devGroup->owl,1,&geom,OPTIX_BUILD_FLAG_ALLOW_UPDATE);
      owlGroupBuildAccel(group);
      volume->generatedGroups.push_back(group);

      // (re-)set this to valid AFTER initial build
      owlGeomSetBuffer(geom,"mcGrid.majorants",mcGrid.majorantsBuffer);
    }
    
    if (volume->xf.domain.lower < volume->xf.domain.upper) {
      owlGeomSet2f(geom,"xf.domain",volume->xf.domain.lower,volume->xf.domain.upper);
    } else {
      owlGeomSet2f(geom,"xf.domain",mesh->worldBounds.lower.w,mesh->worldBounds.upper.w);
    }
    owlGeomSet1f(geom,"xf.baseDensity",volume->xf.baseDensity);
    owlGeomSet1i(geom,"xf.numValues",(int)volume->xf.values.size());
    PING; PRINT(volume->xf.values.size());
    owlGeomSetBuffer(geom,"xf.values",volume->xf.valuesBuffer);

    std::cout << "refitting ... umesh mc geom" << std::endl;
    owlGroupRefitAccel(volume->generatedGroups[0]);
  }

  // template struct UMeshMCAccelerator<UMeshQCSampler>;
  template struct UMeshMCAccelerator<UMeshCUBQLSampler>;
}
