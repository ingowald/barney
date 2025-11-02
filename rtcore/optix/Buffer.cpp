// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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

    Buffer::~Buffer()
    {
      owlBufferRelease(owl);
    }
    
    void *Buffer::getDD() const
    { return (void*)owlBufferGetPointer(owl,0); }
    
  }
}
