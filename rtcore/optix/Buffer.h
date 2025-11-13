// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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
      virtual ~Buffer();
      void *getDD() const;

      void resize(size_t newNumBytes);
      
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
