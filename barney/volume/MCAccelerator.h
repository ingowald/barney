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
  struct MCAccelerator : public VolumeAccel
  {
    template<typename SampleableField>
    struct DD : public SampleableVolumeAccel::DD<SampleableField> {
      using SampleableVolumeAccel::DD<SampleableField>::field;
      
      static void addVarDecls(std::vector<OWLVarDecl> &vars,size_t base);

      inline __device__ box3f getCellBounds(vec3i cellID) const;
      inline __device__ box3f getCellBounds(int linearID) const;
      
      /*! our own macro-cell grid to be traversed */
      MCGrid::DD                mcGrid;
    };

    MCAccelerator(Volume *volume);
    
    MCGrid       mcGrid;
  };


  // ==================================================================
  // INLINE IMPLEMENTATION SECTION
  // ==================================================================
  
  // template<typename SampleableField>
  MCAccelerator// <SampleableField>
  ::MCAccelerator(Volume *volume)
    : VolumeAccel(volume),
      // sampler(field),
      mcGrid(devGroup)
  {}

  template<typename SampleableField>
  inline __device__
  box3f MCAccelerator::DD<SampleableField>::getCellBounds(vec3i cellID) const 
  {
    box3f bounds;
    bounds.lower
      = lerp(getBox(this->field.worldBounds),
             vec3f(cellID)*rcp(vec3f(mcGrid.dims)));
    bounds.upper
      = lerp(getBox(this->field.worldBounds),
             vec3f(cellID+vec3i(1))*rcp(vec3f(mcGrid.dims)));
    return bounds;
  }
      
  template<typename SampleableField>
  inline __device__
  box3f MCAccelerator::DD<SampleableField>::getCellBounds(int linearID) const 
  {
    vec3i dims = mcGrid.dims;
    vec3i cellID(linearID % dims.x,
                 (linearID / dims.x) % dims.y,
                 linearID / (dims.x*dims.y));
    return getCellBounds(cellID);
  }
      
}

