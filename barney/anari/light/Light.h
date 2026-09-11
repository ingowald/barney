// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "anari/Object.h"

namespace BARNEY_NS {
  namespace anari {

    struct Light : public BARNEY_NS::anari::Object
    {
      Light(BarneyGlobalState *s);
      ~Light() override;

      static Light *createInstance(std::string_view subtype,
                                   BarneyGlobalState *state);

      void markFinalized() override;
      virtual void commitParameters() override;
      void finalize() override;

      BNLight getBarneyLight();

    protected:
      virtual const char *bnSubtype() const = 0;
      virtual void setBarneyParameters() = 0;

      math::float3 m_color{1.f, 1.f, 1.f};

      BNLight m_bnLight{nullptr};
    };

  }
} // namespace BANARI_NS

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::Light *, ANARI_LIGHT);
