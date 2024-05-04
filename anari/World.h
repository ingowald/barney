// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Array.h"
#include "Instance.h"

namespace barney_device {

struct World : public Object
{
  World(BarneyGlobalState *s);
  ~World() override;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags) override;

  void commit() override;

  BNModel makeCurrent();

 private:
  bool isCurrent() const;
  void buildBarneyModel();

  helium::CommitObserverPtr<ObjectArray> m_zeroSurfaceData;
  helium::CommitObserverPtr<ObjectArray> m_zeroVolumeData;
  helium::CommitObserverPtr<ObjectArray> m_zeroLightData;
  helium::CommitObserverPtr<ObjectArray> m_instanceData;

  helium::IntrusivePtr<Group> m_zeroGroup;
  helium::IntrusivePtr<Instance> m_zeroInstance;

  std::vector<Instance *> m_instances;

  BNModel m_barneyModel{nullptr};
  helium::TimeStamp m_lastBarneyModelBuild{0};
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::World *, ANARI_WORLD);
