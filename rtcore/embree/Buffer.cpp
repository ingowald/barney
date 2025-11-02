// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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
