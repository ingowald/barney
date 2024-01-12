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

#include "barney/volume/MCAccelerator.h"
#include "barney/umesh/CUBQLFieldSampler.h"

namespace barney {

  /*! a macrocell accelerator built over umeshes */
  template<typename FieldSampler>
  struct UMeshMCAccelerator : public MCAccelerator<FieldSampler>
  {
    using MCAccelerator<FieldSampler>::mcGrid;
    using MCAccelerator<FieldSampler>::volume;

    struct DD : public MCAccelerator<FieldSampler>::DD {
      using MCAccelerator<FieldSampler>::DD::sampleAndMap;
      // UMeshField::DD mesh;
    };
    
    UMeshMCAccelerator(UMeshField *mesh, Volume *volume)
      : MCAccelerator<FieldSampler>(mesh,volume),
        mesh(mesh)
    {}
    static OWLGeomType createGeomType(DevGroup *devGroup);
    
    
    // void buildMCs() override;
    void build(bool full_rebuild) override;

    OWLGeom geom = 0;
      
    UMeshField *const mesh;
  };

  typedef UMeshMCAccelerator<CUBQLFieldSampler> UMeshAccel_MC_CUBQL;
  
}
