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

#include "barney/volume/Volume.h"
#include "barney/volume/ScalarField.h"
#include "barney/DataGroup.h"

namespace barney {

  OWLContext VolumeAccel::getOWL() const
  { return sf->getOWL(); }
  
  Volume::Volume(DevGroup *devGroup,
                 ScalarField::SP sf)
    : devGroup(devGroup), sf(sf), xf(devGroup)
  {
    accel = sf->createAccel(this);
  }

  
  /*! (re-)build the accel structure for this volume, probably after
    changes to transfer functoin (or later, scalar field) */
  void Volume::build(bool full_rebuild)
  {
    PING;
    assert(accel);
    PING;
    accel->build(full_rebuild);
    devGroup->sbtDirty = true;
  }

  std::vector<OWLVarDecl> VolumeAccel::getVarDecls(uint32_t baseOfs)
  {
    return volume->xf.getVarDecls(baseOfs);
  }
  
  // void VolumeAccel::setVariables(OWLGeom geom, bool firstTime)
  // {
  //   volume->xf.setVariables(geom,firstTime);
  //   sf->setVariables(geom,firstTime);
  // }
    
  
  // std::vector<OWLVarDecl> ScalarField::getVarDecls(uint32_t baseOfs)
  // {
  //   return
  //     {
  //      { "worldBounds.lower", OWL_FLOAT4, baseOfs+OWL_OFFSETOF(DD,worldBounds.lower) },
  //      { "worldBounds.upper", OWL_FLOAT4, baseOfs+OWL_OFFSETOF(DD,worldBounds.upper) }
  //     };
  // }
  
}
