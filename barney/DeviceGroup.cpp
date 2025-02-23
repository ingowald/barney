// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

namespace BARNEY_NS {

  RTC_IMPORT_COMPUTE1D(setTileCoords);
  RTC_IMPORT_COMPUTE1D(compressTiles);
  RTC_IMPORT_COMPUTE1D(unpackTiles);
    
  RTC_IMPORT_COMPUTE1D(toneMap);
  RTC_IMPORT_COMPUTE1D(toFixed8);
  RTC_IMPORT_COMPUTE1D(generateRays);
  RTC_IMPORT_COMPUTE1D(shadeRays);

  // umesh related:
  RTC_IMPORT_COMPUTE1D(umeshCreateElements);
  RTC_IMPORT_COMPUTE1D(umeshRasterElements);
  RTC_IMPORT_COMPUTE1D(umeshReorderElements);
  RTC_IMPORT_COMPUTE1D(umeshComputeElementBBs);

  RTC_IMPORT_TRACE2D(traceRays,traceRays);

  
  GeomTypeRegistry::GeomTypeRegistry(rtc::Device *device)
    : device(device)
  {}
  
  rtc::GeomType *GeomTypeRegistry::get(GeomTypeCreationFct callBack,
                                       const void *cbData)
  {
    if (geomTypes.find(callBack) == geomTypes.end()) {
      geomTypes[callBack] = callBack(device,cbData);
    }
    return geomTypes[callBack];
  }

  void Device::syncPipelineAndSBT()
  {
    if (programsDirty) {
      if (Context::logging())
        std::cout << "rebuilding ray tracing programs and pipeline..." << std::endl;
      rtc->buildPipeline();
      programsDirty = false;
    }
    if (sbtDirty) {
      rtc->buildSBT();
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
    rayQueue = new RayQueue(this);
    setTileCoords
      // = rtc->createCompute("setTileCoords");
      = createCompute_setTileCoords(rtc);
    compressTiles
      // = rtc->createCompute("compressTiles");
      = createCompute_compressTiles(rtc);
    unpackTiles
      = createCompute_unpackTiles(rtc);
    
    toneMap
      = createCompute_toneMap(rtc);
    toFixed8
      = createCompute_toFixed8(rtc);
    generateRays
      = createCompute_generateRays(rtc);
    shadeRays
      = createCompute_shadeRays(rtc);

    // umesh related:
    umeshCreateElements 
      = createCompute_umeshCreateElements(rtc);
    umeshRasterElements 
      = createCompute_umeshRasterElements(rtc);
    umeshReorderElements 
      = createCompute_umeshReorderElements(rtc);
    umeshComputeElementBBs
      = createCompute_umeshComputeElementBBs(rtc);
      
    traceRays
      = createTrace_traceRays(rtc);
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
