// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/embree/Device.h"

namespace rtc {
  namespace embree {

    struct Buffer
    {
      Buffer(Device *device,size_t numBytes,const void *initMem);
      virtual ~Buffer();

      void upload(const void *hostPtr,
                  size_t numBytes,
                  size_t ofs = 0);

      void resize(size_t newNumBytes);

      void *getDD() const;
      void *mem = 0;
    };

  }
}
