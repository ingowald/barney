// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace barney_device {

struct Renderer : public Object
{
  Renderer(BarneyGlobalState *s);
  ~Renderer() override;
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Renderer *, ANARI_RENDERER);
