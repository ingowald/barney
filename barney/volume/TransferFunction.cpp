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

#include "barney/volume/TransferFunction.h"

namespace BARNEY_NS {

  TransferFunction::TransferFunction(Context *context,
                                     const DevGroup::SP &devices)
    : SlottedObject(context,devices)
  {
    perLogical.resize(devices->numLogical);
    domain = { 0.f,1.f };
    values = { vec4f(1.f), vec4f(1.f) };
    baseDensity  = 1.f;
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      pld->valuesBuffer
        = device->rtc->createBuffer(sizeof(rtc::float4)*values.size(),
                               values.data());
    }
  }

  TransferFunction::PLD *TransferFunction::getPLD(Device *device)
  { return &perLogical[device->contextRank()]; } 

  void TransferFunction::set(const range1f &domain,
                             const std::vector<vec4f> &values,
                             float baseDensity)
  {
    this->domain = domain;
    this->baseDensity = baseDensity;
    this->values = values;
    
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      device->rtc->freeBuffer(pld->valuesBuffer);
      pld->valuesBuffer
        = device->rtc->createBuffer(sizeof(rtc::float4)*values.size(),
                               values.data());
    }
  }
  
  /*! get cuda-usable device-data for given device ID (relative to
    devices in the devgroup that this gris is in */
  TransferFunction::DD TransferFunction::getDD(Device *device) 
  {
    TransferFunction::DD dd;
    
    dd.values = (rtc::float4*)getPLD(device)->valuesBuffer->getDD();
    dd.domain = domain;
    dd.baseDensity = baseDensity;
    dd.numValues = (int)values.size();

    return dd;
  }
    
}
