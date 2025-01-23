// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

namespace barney {

  Volume::PLD *Volume::getPLD(Device *device)
  { return &perLogical[device->contextRank]; }

  // OWLContext VolumeAccel::getOWL() const
  // { return sf->getOWL(); }

  //   void fillDD(Vo::DD &dd,
  //               rtc::Device *device);
  // // void VolumeAccel::setVariables(OWLGeom geom)
  // {
  //   // owlGeomSet3fv(geom,"domain,lower",&volume->domain.lower.x);
  //   // owlGeomSet3fv(geom,"domain,upper",&volume->domain.upper.x);
  //   getXF()->setVariables(geom);
  // }

  inline ScalarField::SP assertNotNull(const ScalarField::SP &s)
  { assert(s); return s; }
  
  inline ScalarField *assertNotNull(ScalarField *s)
  { assert(s); return s; }
  
  Volume::Volume(ScalarField::SP sf)
    : SlottedObject(sf->context,sf->devices),
      sf(sf),
      xf(sf->context,sf->devices)
  {
    accel = sf->createAccel(this);
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
