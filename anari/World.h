// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Array.h"
#include "Instance.h"

namespace tally_device {

struct World : public Object
{
  World(TallyGlobalState *s);
  ~World() override;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags) override;

  void commit() override;

  TallyModel::SP makeCurrent();

 private:
  void buildTallyModel();

  helium::ChangeObserverPtr<ObjectArray> m_zeroSurfaceData;
  helium::ChangeObserverPtr<ObjectArray> m_zeroVolumeData;
  helium::ChangeObserverPtr<ObjectArray> m_zeroLightData;
  helium::ChangeObserverPtr<ObjectArray> m_instanceData;

  helium::IntrusivePtr<Group> m_zeroGroup;
  helium::IntrusivePtr<Instance> m_zeroInstance;

  std::vector<Instance *> m_instances;

  TallyModel::SP m_tallyModel;
  helium::TimeStamp m_lastTallyModelBuild{0};
};

} // namespace tally_device

TALLY_ANARI_TYPEFOR_SPECIALIZATION(tally_device::World *, ANARI_WORLD);
