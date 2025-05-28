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

  bool Tether::allDevicesPresent()
  {
    for (auto dev : devices)
      if (dev == 0) return false;
    return true;
  }

  TetheredModel *Tether::getModel(int uniqueID)
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto &pair = activeModels[uniqueID];
    if (!pair.second) {
      pair.second = std::make_shared<TetheredModel>();
      pair.second->model = bnModelCreate(context);
    }
    pair.first++;
    return pair.second.get();
  }
  
  void Tether::releaseModel(int uniqueID)
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto &tm = activeModels[uniqueID];
    if (--tm.first == 0) {
      if (tm.second->model)
        bnRelease(tm.second->model);
      activeModels.erase(activeModels.find(uniqueID));
    }
           
  }
  
} // namespace barney_device
