// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#define ANARI_TALLY_MATH_DEFINITIONS 1

#include "TallyGlobalState.h"
#include "Frame.h"

namespace tally_device {

TallyGlobalState::TallyGlobalState(ANARIDevice d)
    : helium::BaseGlobalDeviceState(d)
{}

void TallyGlobalState::waitOnCurrentFrame() const
{
  if (currentFrame)
    currentFrame->wait();
}

void TallyGlobalState::markSceneChanged()
{
  objectUpdates.lastSceneChange = helium::newTimeStamp();
}

} // namespace tally_device
