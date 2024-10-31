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
// #include "barney/material/Globals.h"
// #include "barney/render/DeviceMaterial.h"
#include "barney/render/Sampler.h"

namespace barney {
  namespace render {
    
    struct MaterialRegistry {
      typedef std::shared_ptr<MaterialRegistry> SP;
    
      MaterialRegistry(DevGroup::SP devGroup);
      virtual ~MaterialRegistry();
      
      int allocate();
      void release(int nowReusableID);
      void grow();

      void setMaterial(int materialID, const DeviceMaterial &, int deviceID);
      const DeviceMaterial *getPointer(int owlDeviceID) const;
    
      int numReserved = 0;
      int nextFree = 0;
    
      std::stack<int> reusableIDs;
      OWLBuffer       buffer = 0;
      DevGroup::SP    devGroup;
    };

  }
}
