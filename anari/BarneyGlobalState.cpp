// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#define ANARI_BARNEY_MATH_DEFINITIONS 1

#include "BarneyGlobalState.h"
#include "Frame.h"

namespace barney_device {

BarneyGlobalState::BarneyGlobalState(ANARIDevice d)
    : helium::BaseGlobalDeviceState(d)
{}

void BarneyGlobalState::markSceneChanged()
{
  objectUpdates.lastSceneChange = helium::newTimeStamp();
}

} // namespace barney_device