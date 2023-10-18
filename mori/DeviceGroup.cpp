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

#include "mori/DeviceGroup.h"

namespace mori {

  Device::Device(int cudaID,
                 int globalIndex,
                 int globalIndexStep)
    : cudaID(cudaID),
      owlContext(owlContextCreate(&cudaID,1)),
      nonLaunchStream(owlContextGetStream(owlContext,0)),
      globalIndex(globalIndex),
      globalIndexStep(globalIndexStep)
  {
  }

  OWLGeomType
  Device::getOrCreateGeomTypeFor(const std::string &geomTypeString,
                                 OWLGeomType (*createOnce)(Device *))
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    OWLGeomType gt = geomTypes[geomTypeString];
    if (gt)
      return gt;
    
    gt = geomTypes[geomTypeString] = createOnce(this);
    return gt;
  }
  
  
  DevGroup::DevGroup(const std::vector<Device::SP> &devices)
    : devices(devices)
  {}
  
}
