// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Group.h"

namespace barney_device {

Group::Group(BarneyGlobalState *s) : Object(ANARI_GROUP, s) {}

Group::~Group()
{
  cleanup();
}

void Group::commit()
{
  cleanup();

  m_surfaceData = getParamObject<ObjectArray>("surface");
  m_volumeData = getParamObject<ObjectArray>("volume");

  if (m_surfaceData)
    m_surfaceData->addCommitObserver(this);
  if (m_volumeData)
    m_volumeData->addCommitObserver(this);
}

void Group::markCommitted()
{
  deviceState()->markSceneChanged();
  Object::markCommitted();
}

  BNGroup Group::makeBarneyGroup(BNModel model, int slot) const
{
  std::vector<BNGeom> barneyGeometries;
  std::vector<Surface *> surfaces;
  std::vector<BNVolume> barneyVolumes;
  std::vector<Volume *> volumes;

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
    barneyGeometries.push_back(s->makeBarneyGeom(model,slot));

  if (m_volumeData) {
    std::for_each(
        m_volumeData->handlesBegin(), m_volumeData->handlesEnd(), [&](auto *o) {
          auto *v = (Volume *)o;
          if (v && v->isValid())
            volumes.push_back(v);
        });
  }

  for (auto v : volumes)
    barneyVolumes.push_back(v->makeBarneyVolume(model,slot));

  BNGroup bg = bnGroupCreate(model,slot,
      barneyGeometries.data(),
      barneyGeometries.size(),
      barneyVolumes.data(),
      barneyVolumes.size());
  bnGroupBuild(bg);
  return bg;  
}

box3 Group::bounds() const
{
  box3 result;
  result.invalidate();
  if (m_surfaceData) {
    std::for_each(m_surfaceData->handlesBegin(),
        m_surfaceData->handlesEnd(),
        [&](auto *o) {
          auto *s = (Surface *)o;
          if (s && s->isValid())
            result.insert(s->geometry()->bounds());
        });
  }

  if (m_volumeData) {
    std::for_each(
        m_volumeData->handlesBegin(), m_volumeData->handlesEnd(), [&](auto *o) {
          auto *v = (Volume *)o;
          if (v && v->isValid())
            result.insert(v->bounds());
        });
  }
  return result;
}

void Group::cleanup()
{
  if (m_surfaceData)
    m_surfaceData->removeCommitObserver(this);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Group *);
