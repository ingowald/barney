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
#include "barney/Context.h"

namespace barney {

  extern "C" char traceRays_ptx[];

  // Device::Device(rtc::Device *rtc,
  //                int contextRank,
  //                int contextSize,
  //                // int cudaID,
  //                // int owlID,
  //                int globalIndex,
  //                int globalIndexStep)
  //   : contextRank(contextRank),
  //     contextSize(contextSize),
  //     // cudaID(cudaID),
  //     // owlID(owlID),
  //     rtc(rtc),
  //     // launchStream(devGroup?owlContextGetStream(devGroup->owl,owlID):0),
  //     globalIndex(globalIndex),
  //     globalIndexStep(globalIndexStep),
  //     geomTypes(rtc)
  // {
  // }

  void Device::syncPipelineAndSBT()
  {
    if (programsDirty) {
      if (Context::logging())
        std::cout << "rebuilding owl programs and pipeline..." << std::endl;
      // owlBuildPrograms(owl);
      // owlBuildPipeline(owl);
      // for (auto device : devices)
      //   device->rtc->buildPipeline();
      // devGroup->buildPipeline();
      rtc->buildPipeline();
      programsDirty = false;
    }
    if (sbtDirty) {
      // std::cout << "rebuilding owl sbt..." << std::endl;
      // for (auto device : devices)
      //   device->rtc->buildSBT();
      rtc->buildSBT();
      // devGroup->buildSBT();
      // owlBuildSBT(owl);
      sbtDirty = false;
    }
  }

  DevGroup::DevGroup(const std::vector<Device*> &devices,
                     int numLogical)
    : std::vector<Device *>(devices),
      numLogical(numLogical)
  {}


  Device::Device(rtc::Device *rtc,
                 int contextRank,
                 int contextSize,
                 int globalIndex,
                 int globalIndexStep)
    : contextRank(contextRank),
      contextSize(contextSize),
      globalIndex(globalIndex),
      globalIndexStep(globalIndexStep),rtc(rtc),
      geomTypes(rtc)
  {
    setTileCoords
      = rtc->createCompute("setTileCoords");
    compressTiles
      = rtc->createCompute("compressTiles");
    generateRays
      = rtc->createCompute("generateRays");
    shadeRays
      = rtc->createCompute("shadeRays");
    traceRays
      = rtc->createTrace("traceRays",sizeof(barney::render::OptixGlobals));
  }
    
  
#if 0
  DevGroup::DevGroup(int lmsIdx,
                     const std::vector<int> &contextRanks,
                     int contextSize,
                     const std::vector<int> &gpuIDs,
                     int globalIndex,
                     int globalIndexStep)
    : lmsIdx(lmsIdx)
  {
    auto backend = rtc::Backend::get();
    rtc = backend->createDevGroup(gpuIDs);
    
    for (int localID=0;localID<gpuIDs.size();localID++) {
      assert(localID < rtc->devices.size());
      assert(localID < contextRanks.size());
      devices.push_back
        (std::make_shared<Device>(this,
                                  rtc->devices[localID],
                                  contextRanks[localID],
                                  contextSize,
                                  // gpuIDs[localID],localID,
                                  (int)(globalIndex*gpuIDs.size())+localID,
                                  (int)(globalIndexStep*gpuIDs.size())));
    }

    setTileCoordsKernel
      = rtc->createCompute("setTileCoords");
    compressTilesKernel
      = rtc->createCompute("compressTiles");
    generateRaysKernel
      = rtc->createCompute("generateRays");
    shadeRaysKernel
      = rtc->createCompute("shadeRays");
    traceRaysKernel
      = rtc->createTrace("traceRays",sizeof(barney::render::OptixGlobals));
  }
#endif

  // DevGroup::~DevGroup()
  // {
  //   std::cout << "DEVGROUP DESTROYING context " << (int*)rtc << std::endl;
  //   rtc->destroy();
  //   rtc = nullptr;
  // }
  
}
