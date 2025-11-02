// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/volume/Volume.h"
#include "barney/volume/ScalarField.h"
#include "barney/ModelSlot.h"

namespace BARNEY_NS {

  Volume::PLD *Volume::getPLD(Device *device)
  { return &perLogical[device->contextRank()]; }

  void Volume::setXF(const range1f &domain,
                     const bn_float4 *_values,
                     int numValues,
                     float baseDensity) 
  {
    std::vector<vec4f> values(numValues);
    memcpy(values.data(),_values,numValues*sizeof(*_values));
    xf.set(domain,values,baseDensity);
  }

  bool Volume::set1i(const std::string &member,
                     const int   &value) 
  {
    if (member == "userID") {
      userID = value;
      return true; 
    } 
    
    return false;
  }
  
  inline ScalarField::SP assertNotNull(const ScalarField::SP &s)
  { assert(s); return s; }
  
  inline ScalarField *assertNotNull(ScalarField *s)
  { assert(s); return s; }
  
  Volume::Volume(ScalarField::SP sf)
    : barney_api::Volume(sf->context),
      sf(sf),
      xf((Context*)sf->context,sf->devices),
      devices(sf->devices)
  {
    accel = sf->createAccel(this);
    perLogical.resize(devices->numLogical);
  }

  const TransferFunction *VolumeAccel::getXF() const { return &volume->xf; }
  
  /*! (re-)build the accel structure for this volume, probably after
    changes to transfer functoin (or later, scalar field) */
  void Volume::build(bool full_rebuild)
  {
    assert(accel);
    accel->build(full_rebuild);
    for (auto device : *devices)
      device->sbtDirty = true;
  }

}
