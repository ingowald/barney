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

  DeviceContext::DeviceContext(int gpuID)
    : gpuID(gpuID)
  {
    owlContext = owlContextCreate(&gpuID,1);
    stream = owlContextGetStream(owlContext,0);
  }

  OWLGeomType DeviceContext::getOrCreateTypeFor(const std::string &geomTypeString,
                                                OWLGeomType (*createOnce)(DeviceContext *))
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    OWLGeomType gt = geomTypes[geomTypeString];
    if (gt)
      return gt;
    
    gt = geomTypes[geomTypeString] = createOnce(this);
    return gt;
  }
  
  
  DeviceGroup::DeviceGroup(const std::vector<int> &gpuIDs)
  {
    // owl = owlContextCreate((int *)gpuIDs.data(),gpuIDs.size());
  }
  
}
