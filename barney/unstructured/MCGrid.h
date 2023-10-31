// ======================================================================== //
// Copyright 2022++ Ingo Wald                                               //
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
#include "barney/unstructured/TransferFunction.h"

namespace barney {
  
  struct MCGrid {
    struct DD {
      float   *majorants;
      range1f *scalarRanges;
      vec3i    dims;
    };

    MCGrid(DevGroup *devGroup);
    
    /*! get cuda-usable device-data for given device ID (relative to
        devices in the devgroup that this gris is in */
    DD getDD(int devID) const;

    /*! allocate memory for the given grid */
    void resize(vec3i dims);
    
    /*! build *initial* macro-cell grid (ie, the scalar field min/max
      ranges, but not yet the majorants) over a umesh */
    void computeMajorants(TransferFunction *xf);

    inline bool built() const { return (dims != vec3i(0)); }
    
    /* buffer of range1f's, the min/max scalar values per cell */
    OWLBuffer scalarRangesBuffer = 0;
    /* buffer of floats, the actual per-cell majorants */
    OWLBuffer majorantsBuffer = 0;
    vec3i     dims { 0,0,0 };
    DevGroup *const devGroup;
  };
  
}


