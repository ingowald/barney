// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Light.h"
#include "Surface.h"
#include "Volume.h"
// std
#include <vector>

namespace tally_device {

struct Group : public Object
{
  Group(TallyGlobalState *s);
  ~Group() override;

  void commit() override;
  void markCommitted() override;

  TallyGroup::SP makeTallyGroup(TallyModel::SP model, int slot) const;

  box3 bounds() const;

 private:
  helium::ChangeObserverPtr<ObjectArray> m_surfaceData;
  helium::ChangeObserverPtr<ObjectArray> m_volumeData;
  helium::ChangeObserverPtr<ObjectArray> m_lightData;
};

} // namespace tally_device

TALLY_ANARI_TYPEFOR_SPECIALIZATION(tally_device::Group *, ANARI_GROUP);
