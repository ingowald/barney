// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#define ANARI_BARNEY_MATH_DEFINITIONS 1

#include "BarneyGlobalState.h"
#include "Frame.h"

namespace barney_device {

  Tether::~Tether()
  {
    std::cout << "#banari: tether destructing - releasing barney context" << std::endl;
    if (context) { bnContextDestroy(context); context = 0; }
  }

  BarneyGlobalState::BarneyGlobalState(ANARIDevice d)
    : helium::BaseGlobalDeviceState(d)
  {}

  BarneyGlobalState::~BarneyGlobalState()
  {
    std::cout << "#banari: barneyglobalstate destructing - releasing tether" << std::endl;
  }
  
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

  TetheredModel *Tether::getAndRefModel(int uniqueID)
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto &pair = activeModels[uniqueID];
    if (!pair.second) {
      pair.second = std::make_shared<TetheredModel>();
      pair.second->model = bnModelCreate(context);
    }
    pair.first++;
    std::cout << "#banari GETTING model ID " << uniqueID << " coun1 " << pair.first << std::endl;
    return pair.second.get();
  }
  
  void Tether::releaseModel(int uniqueID)
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto &tm = activeModels[uniqueID];
    std::cout << "#banari: releasing model ID " << uniqueID << " count " << tm.first << std::endl;
    if (--tm.first == 0) {
      std::cout << "#banari: tether releases barney model!" << std::endl;
      if (tm.second->model)
        bnRelease(tm.second->model);
      activeModels.erase(activeModels.find(uniqueID));
    }
           
  }
  
} // namespace barney_device
