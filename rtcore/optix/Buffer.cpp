// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

#include "rtcore/optix/Buffer.h"
#include "rtcore/optix/Device.h"

namespace rtc {
  namespace optix {

    Buffer::Buffer(optix::Device *device,
                   size_t size,
                   const void *initData)
      : device(device)
    {
      owl = owlDeviceBufferCreate(device->owl,OWL_BYTE,size,initData);
    }
    
    void *Buffer::getDD() const
    { return (void*)owlBufferGetPointer(owl,0); }
    
  }
}
