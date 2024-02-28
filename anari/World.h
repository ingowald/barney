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

  BNModel barneyModel() const;

  void barneyModelUpdate();

 private:
  void buildBarneyModel();
  void cleanup();

  helium::IntrusivePtr<ObjectArray> m_zeroSurfaceData;
  helium::IntrusivePtr<ObjectArray> m_zeroVolumeData;
  helium::IntrusivePtr<ObjectArray> m_instanceData;

  bool m_addZeroInstance{false};
  helium::IntrusivePtr<Group> m_zeroGroup;
  helium::IntrusivePtr<Instance> m_zeroInstance;

  std::vector<Instance *> m_instances;

  BNModel m_barneyModel{nullptr};
  int     m_barneySlot {-1};
  // int m_barneyDataGroup{nullptr};

  helium::TimeStamp m_lastBarneyModelBuild{0};
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::World *, ANARI_WORLD);
