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
#include "barney/volume/Volume.h"
#include "barney/volume/MCGrid.h"

namespace barney {

  /*! a macro-cell accelerator, built over some
      (template-parameter'ed) type of underlying volume. The volume
      must be able to compute the macro-cells and majorants, and to
      sample; this class will then do the traversal, and provide the
      'glue' to act as a actual barney volume accelerator */
  template<typename FieldSampler>
  struct MCAccelerator : public VolumeAccel
  {
    struct DD {
      inline __device__
      vec4f sampleAndMap(vec3f P, bool dbg=false) const
      { return volume.sampleAndMap(sampler,P,dbg); }

      /*! our own macro-cell grid to be traversed */
      MCGrid::DD                mcGrid;
      
      /*! whatever the field sampler brings in to be able to sample
          the underlying field */
      typename FieldSampler::DD sampler;
      
      /*! the volume's device data that maps field samples to rgba
          values */
      VolumeAccel::DD           volume;
    };

    MCAccelerator(ScalarField *field, Volume *volume);
    // void build() override;
    
    OWLGeom      geom = 0;
    MCGrid       mcGrid;
    FieldSampler sampler;
  };


  // ==================================================================
  // INLINE IMPLEMENTATION SECTION
  // ==================================================================
  
  template<typename FieldSampler>
  MCAccelerator<FieldSampler>::MCAccelerator(ScalarField *field,
                                              Volume *volume)
    : VolumeAccel(field, volume),
      sampler(field),
      mcGrid(devGroup)
  {}

}

