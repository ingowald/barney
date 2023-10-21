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

#include "barney/Volume.h"
#include "barney/DataGroup.h"

namespace barney {

  TransferFunction::TransferFunction(DataGroup *owner,
                                     const range1f &domain,
                                     const std::vector<float4> &values,
                                     float baseDensity)
    : owner(owner),
      domain(domain),
      values(values),
      baseDensity(baseDensity)
  {
    valuesBuffer = owlDeviceBufferCreate(owner->devGroup->owl,
                                         OWL_FLOAT4,
                                         values.size(),
                                         values.data());
  }
  
}
