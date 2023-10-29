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

  template<typename VolumeSampler>
  struct MCAccelerator : public VolumeAccel
  {
    struct DD {
      template<typename FieldSampler>
      inline __device__
      void traceRay(const FieldSampler &sampler, Ray &ray);
      
      MCGrid::DD                 mcGrid;
      typename VolumeSampler::DD sampler;
      TransferFunction::DD       xf;
      // box4f                      worldBoundsOfGrid;
    };

    MCAccelerator(ScalarField *field, Volume *volume);
    
    virtual void buildMCs() = 0;
    
    MCGrid        mcGrid;
    VolumeSampler sampler;
  };

  template<typename VolumeSampler>
  MCAccelerator<VolumeSampler>::MCAccelerator(ScalarField *field,
                                              Volume *volume)
    : VolumeAccel(field, volume),
      sampler(field),
      mcGrid(devGroup)
  {}

  
}

