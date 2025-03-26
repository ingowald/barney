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

#include "barney/volume/Volume.h"
#include "barney/volume/ScalarField.h"
#include "barney/ModelSlot.h"

namespace BARNEY_NS {

  Volume::PLD *Volume::getPLD(Device *device)
  { return &perLogical[device->contextRank]; }

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
