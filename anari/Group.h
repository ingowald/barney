// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Surface.h"
#include "Volume.h"
// std
#include <vector>

namespace barney_device {

struct Group : public Object
{
  Group(BarneyGlobalState *s);
  ~Group() override;

  void commit() override;
  void markCommitted() override;

  BNGroup makeBarneyGroup(BNDataGroup dg) const;

  box3 bounds() const;

private:
  void cleanup();

  helium::IntrusivePtr<ObjectArray> m_surfaceData;
  helium::IntrusivePtr<ObjectArray> m_volumeData;
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Group *, ANARI_GROUP);
