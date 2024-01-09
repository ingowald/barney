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
#include "barney/DataGroup.h"

namespace barney {

  OWLContext ScalarField::getOWL() const
  { return devGroup->owl; }//owner->getOWL(); }
  
  Volume::Volume(DevGroup *devGroup,
                 ScalarField::SP field)
    : devGroup(devGroup), field(field), xf(devGroup)
  {
    accel = field->createAccel(this);
  }

  void VolumeAccel::setVariables(OWLGeom geom)
  {
    field->setVariables(geom);
    getXF()->setVariables(geom);
  }
  
  /*! (re-)build the accel structure for this volume, probably after
    changes to transfer functoin (or later, scalar field) */
  void Volume::build()
  {
    assert(accel);
    accel->build();
    devGroup->sbtDirty = true;
  }

  // void VolumeAccel::setVariables(OWLGeom geom, bool firstTime)
  // {
  //   volume->xf.setVariables(geom,firstTime);
  //   field->setVariables(geom,firstTime);
  // }
    
  
  void ScalarField::DD::addVarDecls(std::vector<OWLVarDecl> &vars,uint32_t base)
  {
    std::vector<OWLVarDecl> mine =
      {
        { "worldBounds.lower", OWL_FLOAT4, base+OWL_OFFSETOF(DD,worldBounds.lower) },
        { "worldBounds.upper", OWL_FLOAT4, base+OWL_OFFSETOF(DD,worldBounds.upper) }
      };
    for (auto var : mine)
      vars.push_back(var);
  }
  
  void ScalarField::setVariables(OWLGeom geom)
  {
    owlGeomSet4fv(geom,"worldBounds.lower",&worldBounds.lower.x);
    owlGeomSet4fv(geom,"worldBounds.upper",&worldBounds.upper.x);
  }
}
