// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Renderer.h"

namespace barney_device {

Renderer::Renderer(BarneyGlobalState *s) : Object(ANARI_RENDERER, s) {}

Renderer::~Renderer() = default;

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Renderer *);
