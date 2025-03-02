// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Group.h"

namespace barney_device {

struct Instance : public Object
{
  Instance(BarneyGlobalState *s);
  ~Instance() override;

  void commitParameters() override;
  void finalize() override;
  void markFinalized() override;

  bool isValid() const override;

  const Group *group() const;

  const BNTransform *barneyTransform() const;

  box3 bounds() const;

 private:
  math::mat4 m_transform;
  BNTransform m_xfm;
  helium::IntrusivePtr<Group> m_group;
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Instance *, ANARI_INSTANCE);
