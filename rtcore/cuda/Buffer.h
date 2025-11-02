// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/cuda/Device.h"

namespace rtc {
  namespace cuda {
    
    struct Device;
  
    struct Buffer {
      Buffer(Device *device,
             size_t numBytes,
             const void *initValues);
      virtual ~Buffer();
      
      void *getDD() const { return d_data; }
      void upload(const void *data, size_t numBytes, size_t offset=0);
      void resize(size_t newNumBytes);
      void *d_data = 0;
      Device *const device;
    };

  }
}
