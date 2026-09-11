// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "BarneyGlobalState.h"
// helium
#include <anari/anari_cpp.hpp>
#include <helium/array/Array1D.h>
#include <helium/array/Array2D.h>
#include <helium/array/Array3D.h>
#include <helium/array/ObjectArray.h>

namespace BARNEY_NS {
  namespace anari {
  
    using ObjectArray = helium::ObjectArray;

#define DEFINE_BARNEY_ARRAY(DIM)                                        \
    struct Array##DIM : public helium::Array##DIM                       \
    {                                                                   \
      Array##DIM(BarneyGlobalState *state,                              \
                 const helium::Array##DIM##MemoryDescriptor &d);        \
      ~Array##DIM() override;                                           \
                                                                        \
      void unmap() override;                                            \
                                                                        \
      BNData barneyData() const;                                        \
                                                                        \
    private:                                                            \
    mutable BNData m_handle{nullptr};                                   \
    };

    DEFINE_BARNEY_ARRAY(1D)
    DEFINE_BARNEY_ARRAY(2D)
    DEFINE_BARNEY_ARRAY(3D)

    BNDataType anariToBarney(anari::DataType type);

    // Inlined definitions ////////////////////////////////////////////////////////

    inline size_t getNumBytes(const helium::Array &arr)
    {
      return arr.totalCapacity() * anari::sizeOf(arr.elementType());
    }

    inline size_t getNumBytes(const helium::IntrusivePtr<Array1D> &arr)
    {
      return arr ? getNumBytes(*arr) : size_t(0);
    }

    inline size_t getNumBytes(const helium::IntrusivePtr<Array2D> &arr)
    {
      return arr ? getNumBytes(*arr) : size_t(0);
    }

    inline size_t getNumBytes(const helium::IntrusivePtr<Array3D> &arr)
    {
      return arr ? getNumBytes(*arr) : size_t(0);
    }

    template <typename T>
    inline size_t getNumBytes(const std::vector<T> &v)
    {
      return v.size() * sizeof(T);
    }

  }
}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::Array1D *, ANARI_ARRAY1D);
BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::Array2D *, ANARI_ARRAY2D);
BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::Array3D *, ANARI_ARRAY3D);

  
