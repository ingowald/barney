// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Group.h"

namespace tally_device {

struct Instance : public Object
{
  Instance(TallyGlobalState *s);
  ~Instance() override;

  void commit() override;
  void markCommitted() override;

  bool isValid() const override;

  const Group *group() const;

  const TallyTransform *tallyTransform() const;

  box3 bounds() const;

 private:
  TallyTransform m_xfm;
  helium::IntrusivePtr<Group> m_group;
};

} // namespace tally_device

TALLY_ANARI_TYPEFOR_SPECIALIZATION(tally_device::Instance *, ANARI_INSTANCE);
