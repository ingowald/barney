// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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
