// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace barney_device {

struct Light : public Object
{
  Light(BarneyGlobalState *s);
  ~Light() override;

  virtual void commit() override;

  static Light *createInstance(std::string_view type, BarneyGlobalState *state);

  virtual void setBarneyParameters(BNLight light) const = 0;

 protected:
  math::float3 m_radiance{1.f, 1.f, 1.f};

  BNLight m_barneyLight = 0;
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct Directional : public Light
{
  Directional(BarneyGlobalState *s);

  void commit() override;

  void setBarneyParameters(BNLight light) const override;

 private:
  math::float3 m_dir{0.f, 0.f, -1.f};
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Light *, ANARI_LIGHT);
