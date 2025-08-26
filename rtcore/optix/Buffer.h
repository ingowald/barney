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

#pragma once

#include "rtcore/optix/Device.h"
#include <owl/owl.h>

namespace rtc {
  namespace optix {
    struct Device;
    
    struct Buffer {
      Buffer(optix::Device *device,
             size_t size,
             const void *initData);
      void *getDD() const;

      void resize(size_t newNumBytes)
      { owlBufferResize(owl,newNumBytes); }
      
      void upload(const void *hostPtr,
                  size_t numBytes,
                  size_t ofs = 0);
      void uploadAsync(const void *hostPtr,
                       size_t numBytes,
                       size_t ofs = 0);

      optix::Device *const device;      
      OWLBuffer owl;
    };

    inline void Buffer::uploadAsync(const void *hostPtr,
                                    size_t numBytes,
                                    size_t ofs)
    {
      device->copyAsync(((uint8_t*)getDD())+ofs,hostPtr,numBytes);
    }
    inline void Buffer::upload(const void *hostPtr,
                               size_t numBytes,
                               size_t ofs)
    {
      device->copyAsync(((uint8_t*)getDD())+ofs,hostPtr,numBytes);
      device->sync();
    }
    
    
  }
}
