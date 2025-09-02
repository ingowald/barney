// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace barney_device {

struct Light : public Object
{
  Light(BarneyGlobalState *s);
  ~Light() override;

  static Light *createInstance(
      std::string_view subtype, BarneyGlobalState *state);

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

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Light *, ANARI_LIGHT);
