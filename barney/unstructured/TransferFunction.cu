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

#include "barney/unstructured/TransferFunction.h"

namespace barney {

  TransferFunction::TransferFunction(DevGroup *devGroup)
    : devGroup(devGroup)
  {
    domain = { 0.f,0.f };
    values = { vec4f(1.f), vec4f(1.f) };
    baseDensity  = 1.f;
    valuesBuffer = owlDeviceBufferCreate(devGroup->owl,
                                         OWL_FLOAT4,
                                         values.size(),
                                         values.data());
  }
  
  void TransferFunction::set(const range1f &domain,
                             const std::vector<vec4f> &values,
                             float baseDensity)
  {
    this->domain = domain;
    this->baseDensity = baseDensity;
    this->values = values;
    owlBufferResize(this->valuesBuffer,this->values.size());
    owlBufferUpload(this->valuesBuffer,this->values.data());
  }

    /*! get cuda-usable device-data for given device ID (relative to
        devices in the devgroup that this gris is in */
  TransferFunction::DD TransferFunction::getDD(int devID) const
  {
    TransferFunction::DD dd;

    dd.values = (float4*)owlBufferGetPointer(valuesBuffer,devID);
    dd.domain = domain;
    dd.baseDensity = baseDensity;
    dd.numValues = values.size();

    return dd;
  }
    
  
}
