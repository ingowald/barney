// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

#include "barney/DeviceGroup.h"
#include "barney/DeviceContext.h"
#include "barney/render/OptixGlobals.h"

namespace barney {

  extern "C" char traceRays_ptx[];

  Device::Device(DevGroup *devGroup,
                 int cudaID,
                 int owlID,
                 int globalIndex,
                 int globalIndexStep)
    : cudaID(cudaID),
      owlID(owlID),
      devGroup(devGroup),
      launchStream(devGroup?owlContextGetStream(devGroup->owl,owlID):0),
      globalIndex(globalIndex),
      globalIndexStep(globalIndexStep)
  {
  }

  void DevGroup::update()
  {
    if (programsDirty) {
      std::cout << "rebuilding owl programs and pipeline..." << std::endl;
      owlBuildPrograms(owl);
      owlBuildPipeline(owl);
      programsDirty = false;
    }
    if (sbtDirty) {
      std::cout << "rebuilding owl sbt..." << std::endl;
      owlBuildSBT(owl);
      sbtDirty = false;
    }
  }

  DevGroup::DevGroup(int lmsIdx,
                     const std::vector<int> &gpuIDs,
                     int globalIndex,
                     int globalIndexStep)
    : lmsIdx(lmsIdx)
  {
    owl = owlContextCreate((int*)gpuIDs.data(),(int)gpuIDs.size());
    std::cout << "DEVGROUP created owl " << (int*)owl << std::endl;
    OWLVarDecl args[]
      = {
      { nullptr }
    };
    OWLModule module = owlModuleCreate(owl,traceRays_ptx);
    rg = owlRayGenCreate(owl,module,"traceRays",0,args,-1);

    owlBuildPrograms(owl);

    for (int localID=0;localID<gpuIDs.size();localID++)
      devices.push_back
        (std::make_shared<Device>(this,gpuIDs[localID],localID,
                                  globalIndex*gpuIDs.size()+localID,
                                  globalIndexStep*gpuIDs.size()));

    OWLVarDecl params[]
      = {
      { "world", OWL_GROUP, OWL_OFFSETOF(render::OptixGlobals, world) },
      { "materials", OWL_BUFPTR, OWL_OFFSETOF(render::OptixGlobals, materials) },
      { "samplers", OWL_BUFPTR, OWL_OFFSETOF(render::OptixGlobals, samplers) },
      { "rays",  OWL_RAW_POINTER, OWL_OFFSETOF(render::OptixGlobals,rays) },
      { "numRays",  OWL_INT, OWL_OFFSETOF(render::OptixGlobals,numRays) },
      { nullptr }
    };
    lp = owlParamsCreate(owl,
                         sizeof(render::OptixGlobals),
                         params,
                         -1);
  }

  DevGroup::~DevGroup()
  {
    std::cout << "DEVGROUP DESTROYING context " << (int*)owl << std::endl;
    owlContextDestroy(owl);
    owl = 0;
  }
  
}
