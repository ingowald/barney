// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/DeviceGroup.h"
#include "barney/render/OptixGlobals.h"
#include "barney/Context.h"
#include "barney/render/RayQueue.h"

namespace BARNEY_NS {

  RTC_IMPORT_COMPUTE1D(generateRays);
  RTC_IMPORT_COMPUTE1D(shadeRays);

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
                 const WorkerTopo *topo,
                 int localRank)
    : rtc(rtc),
      geomTypes(rtc),
      topo(topo),
      _localRank(localRank),
      _globalRank(topo->myOffset+localRank)
  {
    assert(_localRank == topo->allDevices[_globalRank].local); 
    rayQueue = new RayQueue(this);
    traceRays
      = createTrace_traceRays(rtc);
  }

  Device::~Device()
  {
    delete rtc;
  }

  int Device::globalRank() const
  { return _globalRank; }
  
  int Device::globalSize() const
  { return (int)topo->allDevices.size(); }
  
  int Device::localRank() const
  { return _localRank; }
  
  int Device::localSize() const
  { return topo->myCount; }
  
  int Device::worldRank() const
  { return topo->allDevices[_globalRank].worldRank; }
  
  // DEPRECATED!
  int Device::contextRank() const
  { return localRank(); }
  
}
