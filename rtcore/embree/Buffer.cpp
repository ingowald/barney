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

#include "rtcore/embree/Buffer.h"

namespace rtc {
  namespace embree {

    Buffer::Buffer(Device *device,
                   size_t numBytes,
                   const void *initMem)
    {
      mem = numBytes?malloc(numBytes):nullptr;
      if (initMem)
        memcpy(mem,initMem,numBytes);
    }
    
    Buffer::~Buffer()
    {
      if (mem) free(mem);
    }

    void Buffer::resize(size_t numBytes)
    {
      if (mem) free(mem);
      mem = numBytes?malloc(numBytes):nullptr;
    }
    
    void *Buffer::getDD() const
    {
      return mem;
    }

    void Buffer::upload(const void *hostPtr,
                        size_t numBytes,
                        size_t ofs)
    {
      memcpy(((uint8_t*)mem) + ofs,hostPtr,numBytes);
    }
      
  }
}
