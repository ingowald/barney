// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "light/Light.h"
#include "Surface.h"
#include "Volume.h"
// std
#include <vector>

namespace barney_device {

  struct Group : public Object
  {
    Group(BarneyGlobalState *s);
    ~Group() override;

    void commitParameters() override;
    void markFinalized() override;

    BNGroup makeBarneyGroup() const;

    box3 bounds() const;

  private:
    helium::ChangeObserverPtr<ObjectArray> m_surfaceData;
    helium::ChangeObserverPtr<ObjectArray> m_volumeData;
    helium::ChangeObserverPtr<ObjectArray> m_lightData;
  };

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Group *, ANARI_GROUP);
