// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "BarneyGlobalState.h"
// helium
#include <helium/array/Array1D.h>
#include <helium/array/Array2D.h>
#include <helium/array/Array3D.h>
#include <helium/array/ObjectArray.h>

namespace barney_device {

using Array1D = helium::Array1D;
using Array2D = helium::Array2D;
using Array3D = helium::Array3D;
using ObjectArray = helium::ObjectArray;

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

} // namespace barney_device
