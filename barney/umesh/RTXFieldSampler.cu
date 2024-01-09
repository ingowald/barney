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

#include "barney/umesh/RTXFieldSampler.h"

namespace barney {
#if 0    
  extern "C" char RTXObjectSpace_ptx[];
  
  OWLGeomType RTXFieldSampler::createGeomType(DevGroup *devGroup)
  {
    static std::mutex mutex;
    static std::map<DevGroup*,OWLGeomType> alreadyCreatedGTs;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto it = alreadyCreatedGTs.find(devGroup);
      if (it != alreadyCreatedGTs.end()) return it->second;
    }
    
    std::cout << OWL_TERMINAL_GREEN
              << "creating 'RTXFieldSampler' geometry type"
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
      (devGroup->owl,RTXFieldSampler_ptx);
    OWLGeomType gt = owlGeomTypeCreate
      (devGroup->owl,OWL_GEOM_USER,sizeof(UMeshField::DD),
       params,-1);
    std::vector<OWLVarDecl> params;
    UMeshField::addVarDecls(params,0);
    owlGeomTypeSetBoundsProg(gt,module,"RTXFieldSamplerBounds");
    owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,"RTXFieldSampler_IS");
    owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"RTXFieldSampler_CH");
    owlBuildPrograms(devGroup->owl);
     
    {
      std::lock_guard<std::mutex> lock(mutex);
      alreadyCreatedGTs[devGroup] = gt;
    }
    return gt;
  }

  RTXFieldSampler::RTXFieldSampler(UMeshField *const mesh)
    : mesh(mesh)
  {}
  

  void RTXFieldSampler::build()
  {
    auto devGroup = mesh->devGroup;
    OWLGeomType gt = createGeomType(devGroup);
  }
#endif
}
