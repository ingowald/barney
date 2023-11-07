// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Group.h"

namespace barney_device {

struct Instance : public Object
{
  Instance(BarneyGlobalState *s);
  ~Instance() override;

  void commit() override;

  bool isValid() const override;

  const Group *group() const;

  const BNTransform *barneyTransform() const;

 private:
  BNTransform m_xfm;
  helium::IntrusivePtr<Group> m_group;
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Instance *, ANARI_INSTANCE);
