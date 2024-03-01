// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Surface.h"

namespace barney_device {

Surface::Surface(BarneyGlobalState *s) : Object(ANARI_SURFACE, s) {}

Surface::~Surface() = default;

void Surface::commit()
{
  m_id = getParam<uint32_t>("id", ~0u);
  m_geometry = getParamObject<Geometry>("geometry");
  m_material = getParamObject<Material>("material");

  if (!m_material) {
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'material' on ANARISurface");
    return;
  }

  if (!m_geometry) {
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'geometry' on ANARISurface");
    return;
  }
}

void Surface::markCommitted()
{
  deviceState()->markSceneChanged();
  Object::markCommitted();
}

const Geometry *Surface::geometry() const
{
  return m_geometry.ptr;
}

const Material *Surface::material() const
{
  return m_material.ptr;
}

BNGeom Surface::makeBarneyGeom(BNModel model, int slot) const
{
  return geometry()->makeBarneyGeometry(
      model, slot, material()->barneyMaterial());
}

size_t Surface::numRequiredGPUBytes() const
{
  return m_geometry ? m_geometry->numRequiredGPUBytes() : size_t(0);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Surface *);
