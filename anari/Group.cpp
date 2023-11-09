// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Group.h"

namespace barney_device {

Group::Group(BarneyGlobalState *s) : Object(ANARI_GROUP, s)
{
  s->objectCounts.groups++;
}

Group::~Group()
{
  cleanup();
  deviceState()->objectCounts.groups--;
}

void Group::commit()
{
  cleanup();

  m_surfaceData = getParamObject<ObjectArray>("surface");

  if (m_surfaceData)
    m_surfaceData->addCommitObserver(this);
}

BNGroup Group::makeBarneyGroup(BNDataGroup dg) const
{
  std::vector<BNGeom> barneyGeometries;
  std::vector<Surface *> surfaces;

  if (m_surfaceData) {
    std::for_each(m_surfaceData->handlesBegin(),
        m_surfaceData->handlesEnd(),
        [&](auto *o) {
          auto *s = (Surface *)o;
          if (s && s->isValid())
            surfaces.push_back(s);
        });
  }

  for (auto s : surfaces)
    barneyGeometries.push_back(s->makeBarneyGeom(dg));

  return bnGroupCreate(
      dg, barneyGeometries.data(), barneyGeometries.size(), nullptr, 0);
}

void Group::cleanup()
{
  if (m_surfaceData)
    m_surfaceData->removeCommitObserver(this);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Group *);
