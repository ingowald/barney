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

  OWLContext ScalarField::getOWL() const
  { return owner->getOWL(); }
  
  Volume::Volume(DataGroup *owner,
                 ScalarField::SP sf)
    : owner(owner), sf(sf)
  {
    xf.domain = { 0.f,0.f };
    xf.values = { vec4f(0.f), vec4f(1.f) };
    xf.baseDensity  = 1.f;
    xf.valuesBuffer = owlDeviceBufferCreate(owner->devGroup->owl,
                                            OWL_FLOAT4,
                                            xf.values.size(),
                                            xf.values.data());
  }

  void Volume::setXF(const range1f &domain,
                     const std::vector<vec4f> &values,
                     float baseDensity)
  {
    xf.domain = domain;
    xf.baseDensity = baseDensity;
    xf.values = values;
    owlBufferResize(xf.valuesBuffer,xf.values.size());
    owlBufferUpload(xf.valuesBuffer,xf.values.data());
  }

}
