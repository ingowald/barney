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

#pragma once

#include "barney/DeviceGroup.h"
// #include "barney/material/device/Material.h"

namespace barney {
  namespace render {
    
    struct Globals {
      Globals(const DevGroup *devGroup);
      struct DD {
        float *MicrofacetDielectricAlbedoTable_dir;
        float *MicrofacetDielectricReflectionAlbedoTable_dir;
        float *MicrofacetDielectricAlbedoTable_avg;
        float *MicrofacetDielectricReflectionAlbedoTable_avg;
      };

      DD getDD(const Device::SP &device) const;

      OWLBuffer MicrofacetDielectricAlbedoTable_dir_buffer = 0;
      OWLBuffer MicrofacetDielectricReflectionAlbedoTable_dir_buffer = 0;
      OWLBuffer MicrofacetDielectricAlbedoTable_avg_buffer = 0;
      OWLBuffer MicrofacetDielectricReflectionAlbedoTable_avg_buffer = 0;
    };
 
  }
}
