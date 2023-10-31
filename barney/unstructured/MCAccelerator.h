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

#pragma once

#include "barney/DeviceGroup.h"
#include "barney/Volume.h"
#include "barney/unstructured/MCGrid.h"

namespace barney {

  template<typename FieldSampler>
  struct MCAccelerator : public VolumeAccel
  {
    struct DD {
      inline __device__
      vec4f sampleAndMap(vec3f P, bool dbg=false) const
      { return volume.sampleAndMap(sampler,P,dbg); }
      
      MCGrid::DD       mcGrid;
      typename FieldSampler::DD sampler;
      VolumeAccel::DD  volume;
    };

    MCAccelerator(ScalarField *field, Volume *volume);
    
    virtual void buildMCs() = 0;
    
    MCGrid       mcGrid;
    FieldSampler sampler;
  };

  template<typename FieldSampler>
  MCAccelerator<FieldSampler>::MCAccelerator(ScalarField *field,
                                              Volume *volume)
    : VolumeAccel(field, volume),
      sampler(field),
      mcGrid(devGroup)
  {}

  
}

