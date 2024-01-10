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

#include "barney/amr/BlockStructuredMCAccelerator.h"

namespace barney {

  extern "C" char BlockStructuredMCAccelerator_ptx[];

  template<typename VolumeSampler>
  OWLGeomType BlockStructuredMCAccelerator<VolumeSampler>::createGeomType(DevGroup *devGroup)
  {
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'BlockStructured_MC_CUBQL' geometry type"
              << OWL_TERMINAL_DEFAULT << std::endl;

    // TODO - this needs to get split into different classes'
    // 'addParams()', so mesh, sampler, tec can all declare and set
    // set theit own
    static OWLVarDecl params[]
      = {
         { "field.blockBounds", OWL_BUFPTR, OWL_OFFSETOF(DD,sampler.field.blockBounds) },
         { "field.blockLevels", OWL_BUFPTR, OWL_OFFSETOF(DD,sampler.field.blockLevels) },
         { "field.blockOffsets", OWL_BUFPTR, OWL_OFFSETOF(DD,sampler.field.blockOffsets) },
         { "field.blockScalars", OWL_BUFPTR, OWL_OFFSETOF(DD,sampler.field.blockScalars) },
         { "field.blockIDs", OWL_BUFPTR, OWL_OFFSETOF(DD,sampler.field.blockIDs) },
         { "field.valueRanges", OWL_BUFPTR, OWL_OFFSETOF(DD,sampler.field.valueRanges) },
         { "field.worldBounds.lower", OWL_FLOAT4, OWL_OFFSETOF(DD,sampler.field.worldBounds.lower) },
         { "field.worldBounds.upper", OWL_FLOAT4, OWL_OFFSETOF(DD,sampler.field.worldBounds.upper) },
         { "field.numBlocks", OWL_INT, OWL_OFFSETOF(DD,sampler.field.numBlocks) },
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
      (devGroup->owl,BlockStructuredMCAccelerator_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(DD),
       params,-1);
    owlGeomTypeSetBoundsProg(gt,module,"BlockStructured_MC_CUBQL_Bounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"BlockStructured_MC_CUBQL_Isec");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"BlockStructured_MC_CUBQL_CH");
    owlBuildPrograms(devGroup->owl);
    
    return gt;
  }

  template<typename VolumeSampler>
  void BlockStructuredMCAccelerator<VolumeSampler>::build()
  {
    auto devGroup = field->devGroup;

    BARNEY_CUDA_SYNC_CHECK();
    this->field->buildMCs(mcGrid);
    mcGrid.computeMajorants(&volume->xf);
    
    BARNEY_CUDA_SYNC_CHECK();
    this->sampler.build();
    BARNEY_CUDA_SYNC_CHECK();

    if (volume->generatedGroups.empty()) {
      std::string gtTypeName = "BlockStructured_MC_CUBQL";
      OWLGeomType gt = devGroup->getOrCreateGeomTypeFor
        (gtTypeName,createGeomType);
      geom
        = owlGeomCreate(devGroup->owl,gt);
      vec3i dims = mcGrid.dims;
      int primCount = dims.x*dims.y*dims.z;
      PING; PRINT(dims); PRINT(primCount);
      owlGeomSetPrimCount(geom,primCount);

      // ------------------------------------------------------------------      
      owlGeomSet3iv(geom,"mcGrid.dims",&mcGrid.dims.x);
      // intentionally set to null for first-time build
      owlGeomSetBuffer(geom,"mcGrid.majorants",0);
      
      // ------------------------------------------------------------------
      owlGeomSet4fv(geom,"field.worldBounds.lower",&field->worldBounds.lower.x);
      owlGeomSet4fv(geom,"field.worldBounds.upper",&field->worldBounds.upper.x);
      owlGeomSetBuffer(geom,"field.blockBounds",field->blockBoundsBuffer);
      owlGeomSetBuffer(geom,"field.blockLevels",field->blockLevelsBuffer);
      owlGeomSetBuffer(geom,"field.blockOffsets",field->blockOffsetsBuffer);
      owlGeomSetBuffer(geom,"field.blockScalars",field->blockScalarsBuffer);
      owlGeomSetBuffer(geom,"field.blockIDs",field->blockIDsBuffer);
      owlGeomSetBuffer(geom,"field.valueRanges",field->valueRangesBuffer);
      
      // ------------------------------------------------------------------      
      owlGeomSetBuffer(geom,"bvhNodes",this->sampler.bvhNodesBuffer);
      
      // ------------------------------------------------------------------      
      
      if (volume->xf.domain.lower < volume->xf.domain.upper) {
        owlGeomSet2f(geom,"xf.domain",volume->xf.domain.lower,volume->xf.domain.upper);
      } else {
        owlGeomSet2f(geom,"xf.domain",field->worldBounds.lower.w,field->worldBounds.upper.w);
      }
      owlGeomSet1f(geom,"xf.baseDensity",volume->xf.baseDensity);
      owlGeomSet1i(geom,"xf.numValues",(int)volume->xf.values.size());
      PING; PRINT(volume->xf.values.size()); PRINT(field->worldBounds);
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
      owlGeomSet2f(geom,"xf.domain",field->worldBounds.lower.w,field->worldBounds.upper.w);
    }
    owlGeomSet1f(geom,"xf.baseDensity",volume->xf.baseDensity);
    owlGeomSet1i(geom,"xf.numValues",(int)volume->xf.values.size());
    PING; PRINT(volume->xf.values.size());
    owlGeomSetBuffer(geom,"xf.values",volume->xf.valuesBuffer);

    std::cout << "refitting ... amr mc geom" << std::endl;
    owlGroupRefitAccel(volume->generatedGroups[0]);
  }

  template struct BlockStructuredMCAccelerator<CUBQLBlockSampler>;
}
