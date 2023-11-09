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

struct Array1D : public helium::Array1D
{
  Array1D(BarneyGlobalState *state, const helium::Array1DMemoryDescriptor &d);
  ~Array1D() override;
};

struct Array2D : public helium::Array2D
{
  Array2D(BarneyGlobalState *state, const helium::Array2DMemoryDescriptor &d);
  ~Array2D() override;
};

struct Array3D : public helium::Array3D
{
  Array3D(BarneyGlobalState *state, const helium::Array3DMemoryDescriptor &d);
  ~Array3D() override;
};

struct ObjectArray : public helium::ObjectArray
{
  ObjectArray(
      BarneyGlobalState *state, const helium::Array1DMemoryDescriptor &d);
  ~ObjectArray() override;
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Array1D *, ANARI_ARRAY1D);
BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Array2D *, ANARI_ARRAY2D);
BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Array3D *, ANARI_ARRAY3D);
BARNEY_ANARI_TYPEFOR_SPECIALIZATION(
    barney_device::ObjectArray *, ANARI_ARRAY1D);
