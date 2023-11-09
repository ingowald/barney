// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#define ANARI_BARNEY_MATH_DEFINITIONS 1

#include "BarneyGlobalState.h"
#include "Frame.h"

namespace barney_device {

BarneyGlobalState::BarneyGlobalState(ANARIDevice d)
    : helium::BaseGlobalDeviceState(d)
{}

void BarneyGlobalState::waitOnCurrentFrame() const
{
  if (currentFrame)
    currentFrame->wait();
}

} // namespace barney_device