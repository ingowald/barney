// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Array.h"

namespace barney_device {

// Array1D //

Array1D::Array1D(
    BarneyGlobalState *state, const helium::Array1DMemoryDescriptor &d)
    : helium::Array1D(state, d)
{
  state->objectCounts.arrays++;
}

Array1D::~Array1D()
{
  asBarneyState(deviceState())->objectCounts.arrays--;
}

// Array2D //

Array2D::Array2D(
    BarneyGlobalState *state, const helium::Array2DMemoryDescriptor &d)
    : helium::Array2D(state, d)
{
  state->objectCounts.arrays++;
}

Array2D::~Array2D()
{
  asBarneyState(deviceState())->objectCounts.arrays--;
}

// Array3D //

Array3D::Array3D(
    BarneyGlobalState *state, const helium::Array3DMemoryDescriptor &d)
    : helium::Array3D(state, d)
{
  state->objectCounts.arrays++;
}

Array3D::~Array3D()
{
  asBarneyState(deviceState())->objectCounts.arrays--;
}

// ObjectArray //

ObjectArray::ObjectArray(
    BarneyGlobalState *state, const helium::Array1DMemoryDescriptor &d)
    : helium::ObjectArray(state, d)
{
  state->objectCounts.arrays++;
}

ObjectArray::~ObjectArray()
{
  asBarneyState(deviceState())->objectCounts.arrays--;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Array1D *);
BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Array2D *);
BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Array3D *);
BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::ObjectArray *);
