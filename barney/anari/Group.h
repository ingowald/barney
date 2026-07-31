// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "anari/light/Light.h"
#include "anari/Surface.h"
#include "anari/Volume.h"
// std
#include <vector>

namespace BARNEY_NS {
  namespace anari {
    
    struct Group : public Object
    {
      Group(BarneyGlobalState *s);
      ~Group() override;

      void commitParameters() override;
      void finalize() override;
      void markFinalized() override;

      BNGroup makeBarneyGroup() const;

      box3 bounds() const;

    private:
      helium::ChangeObserverPtr<ObjectArray> m_surfaceData;
      helium::ChangeObserverPtr<ObjectArray> m_volumeData;
      helium::ChangeObserverPtr<ObjectArray> m_lightData;
    };

  }
}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::Group *, ANARI_GROUP);
