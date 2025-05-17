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
#include "barney/render/OptixGlobals.h"
#include "barney/Context.h"
#include "barney/render/RayQueue.h"

namespace BARNEY_NS {

  RTC_IMPORT_COMPUTE1D(generateRays);
  RTC_IMPORT_COMPUTE1D(shadeRays);

  // umesh related:
  RTC_IMPORT_COMPUTE1D(umeshCreateElements);
  RTC_IMPORT_COMPUTE1D(umeshRasterElements);
  RTC_IMPORT_COMPUTE1D(umeshReorderElements);
  RTC_IMPORT_COMPUTE1D(umeshComputeElementBBs);

  RTC_IMPORT_TRACE2D
  (/*traceRays.cu*/traceRays,
   /*ray gen name */traceRays,
   /*launch params data type*/sizeof(BARNEY_NS::render::OptixGlobals)
   );
  
  
  GeomTypeRegistry::GeomTypeRegistry(rtc::Device *device)
    : device(device)
  {}
  
  rtc::GeomType *GeomTypeRegistry::get(GeomTypeCreationFct callBack)
  {
    if (geomTypes.find(callBack) == geomTypes.end()) {
      geomTypes[callBack] = callBack(device);
    }
    return geomTypes[callBack];
  }

  void Device::syncPipelineAndSBT()
  {
    rtc->buildPipeline();
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
                 int contextSize// ,
                 // int globalIndex,
                 // int globalIndexStep
                 )
    : rtc(rtc),
      geomTypes(rtc)
  {
    rayQueue = new RayQueue(this);

    gpuInNode.rank = contextRank;
    gpuInNode.size = contextSize;
    
    // umesh related:
    umeshCreateElements 
      = createCompute_umeshCreateElements(rtc);
    umeshRasterElements 
      = createCompute_umeshRasterElements(rtc);
    umeshReorderElements 
      = createCompute_umeshReorderElements(rtc);
    umeshComputeElementBBs
      = createCompute_umeshComputeElementBBs(rtc);
      
    generateRays
      = createCompute_generateRays(rtc);
    shadeRays
      = createCompute_shadeRays(rtc);
    traceRays
      = createTrace_traceRays(rtc);
  }
  
}
