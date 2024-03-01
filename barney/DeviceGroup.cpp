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

#include "barney/DeviceGroup.h"
#include "barney/DeviceContext.h"

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

  DevGroup::DevGroup(int ldgID,
                     const std::vector<int> &gpuIDs,
                     int globalIndex,
                     int globalIndexStep)
    : ldgID(ldgID)
  {
    owl = owlContextCreate((int*)gpuIDs.data(),(int)gpuIDs.size());

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
         { "world", OWL_GROUP, OWL_OFFSETOF(DeviceContext::DD, world) },
         { "rays",  OWL_RAW_POINTER, OWL_OFFSETOF(DeviceContext::DD,rays) },
         { "numRays",  OWL_INT, OWL_OFFSETOF(DeviceContext::DD,numRays) },
         { nullptr }
    };
    lp = owlParamsCreate(owl,
                         sizeof(DeviceContext::DD),
                         params,
                         -1);
  }

  DevGroup::~DevGroup()
  {
    owlContextDestroy(owl);
    owl = 0;
  }
  
}
