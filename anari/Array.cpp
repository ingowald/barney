// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "Array.h"

namespace barney_device {

#define DEFINE_BARNEY_ARRAY_IMPL(DIM)                                          \
  Array##DIM::Array##DIM(                                                      \
      BarneyGlobalState *state, const helium::Array##DIM##MemoryDescriptor &d) \
      : helium::Array##DIM(state, d)                                           \
  {}                                                                           \
                                                                               \
  Array##DIM::~Array##DIM()                                                    \
  {                                                                            \
    if (m_handle)                                                              \
      bnRelease(m_handle);                                                     \
  }                                                                            \
                                                                               \
  void Array##DIM::unmap()                                                     \
  {                                                                            \
    helium::Array##DIM::unmap();                                               \
    if (m_handle)                                                              \
      bnDataSet(m_handle, totalCapacity(), data());                            \
  }                                                                            \
                                                                               \
  BNData Array##DIM::barneyData()                                              \
  {                                                                            \
    if (!m_handle) {                                                           \
      auto *state = (BarneyGlobalState *)deviceState();                        \
      int slot = state->slot;                                                  \
      auto context = state->tether->context;                                   \
      m_handle = bnDataCreate(context,                                         \
          slot,                                                                \
          anariToBarney(elementType()),                                        \
          totalCapacity(),                                                     \
          data());                                                             \
    }                                                                          \
    return m_handle;                                                           \
  }

DEFINE_BARNEY_ARRAY_IMPL(1D)
DEFINE_BARNEY_ARRAY_IMPL(2D)
DEFINE_BARNEY_ARRAY_IMPL(3D)

///////////////////////////////////////////////////////////////////////////////

BNDataType anariToBarney(anari::DataType type)
{
  switch (type) {
  case ANARI_UFIXED8:
    return BN_UFIXED8;
  case ANARI_UFIXED8_VEC4:
    return BN_UFIXED8_RGBA;
  case ANARI_UFIXED8_RGBA_SRGB:
    return BN_UFIXED8_RGBA_SRGB;
  case ANARI_UFIXED16:
    return BN_UFIXED16;
  case ANARI_INT8:
    return BN_INT8;
  case ANARI_INT8_VEC2:
    return BN_INT8_VEC2;
  case ANARI_INT8_VEC3:
    return BN_INT8_VEC3;
  case ANARI_INT8_VEC4:
    return BN_INT8_VEC4;
  case ANARI_UINT8:
    return BN_UINT8;
  case ANARI_UINT8_VEC2:
    return BN_UINT8_VEC2;
  case ANARI_UINT8_VEC3:
    return BN_UINT8_VEC3;
  case ANARI_UINT8_VEC4:
    return BN_UINT8_VEC4;
  case ANARI_INT16:
    return BN_INT16;
  case ANARI_INT16_VEC2:
    return BN_INT16_VEC2;
  case ANARI_INT16_VEC3:
    return BN_INT16_VEC3;
  case ANARI_INT16_VEC4:
    return BN_INT16_VEC4;
  case ANARI_UINT16:
    return BN_UINT16;
  case ANARI_UINT16_VEC2:
    return BN_UINT16_VEC2;
  case ANARI_UINT16_VEC3:
    return BN_UINT16_VEC3;
  case ANARI_UINT16_VEC4:
    return BN_UINT16_VEC4;
  case ANARI_INT32:
    return BN_INT32;
  case ANARI_INT32_VEC2:
    return BN_INT32_VEC2;
  case ANARI_INT32_VEC3:
    return BN_INT32_VEC3;
  case ANARI_INT32_VEC4:
    return BN_INT32_VEC4;
  case ANARI_UINT32:
    return BN_UINT32;
  case ANARI_UINT32_VEC2:
    return BN_UINT32_VEC2;
  case ANARI_UINT32_VEC3:
    return BN_UINT32_VEC3;
  case ANARI_UINT32_VEC4:
    return BN_UINT32_VEC4;
  case ANARI_INT64:
    return BN_INT64;
  case ANARI_INT64_VEC2:
    return BN_INT64_VEC2;
  case ANARI_INT64_VEC3:
    return BN_INT64_VEC3;
  case ANARI_INT64_VEC4:
    return BN_INT64_VEC4;
  case ANARI_UINT64:
    return BN_UINT64;
  case ANARI_UINT64_VEC2:
    return BN_UINT64_VEC2;
  case ANARI_UINT64_VEC3:
    return BN_UINT64_VEC3;
  case ANARI_UINT64_VEC4:
    return BN_UINT64_VEC4;
  case ANARI_FLOAT32:
    return BN_FLOAT32;
  case ANARI_FLOAT32_VEC2:
    return BN_FLOAT32_VEC2;
  case ANARI_FLOAT32_VEC3:
    return BN_FLOAT32_VEC3;
  case ANARI_FLOAT32_VEC4:
    return BN_FLOAT32_VEC4;
  case ANARI_FLOAT64:
    return BN_FLOAT64;
  case ANARI_FLOAT64_VEC2:
    return BN_FLOAT64_VEC2;
  case ANARI_FLOAT64_VEC3:
    return BN_FLOAT64_VEC3;
  case ANARI_FLOAT64_VEC4:
    return BN_FLOAT64_VEC4;
  }

  return BN_DATA_UNDEFINED;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Array1D *);
BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Array2D *);
BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Array3D *);
