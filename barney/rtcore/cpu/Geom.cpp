// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rtcore/cpu/Geom.h"

namespace BARNEY_NS {
  namespace rtc {
    
    Geom::Geom(GeomType *type)
      : type(type),
        programData(type->sizeOfProgramData)
    {}
    
    void Geom::setDD(const void *dd)
    {
      memcpy(programData.data(),dd,programData.size());

      uint8_t *ptr = (uint8_t*)programData.data();
      ptr += programData.size();
    }

  }
}

