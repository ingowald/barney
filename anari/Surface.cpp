// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Surface.h"

namespace barney_device {

Surface::Surface(BarneyGlobalState *s) : Object(ANARI_SURFACE, s)
{
  s->objectCounts.groups++;
}

Surface::~Surface()
{
  deviceState()->objectCounts.groups--;
}

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

BNGeom Surface::makeBarneyGeom(BNDataGroup dg) const
{
  return geometry()->makeBarneyGeometry(dg, material()->barneyMaterial());
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Surface *);
