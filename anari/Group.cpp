// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Group.h"
#include <iostream>
#include "common.h"

namespace tally_device {

Group::Group(TallyGlobalState *s)
    : Object(ANARI_GROUP, s),
      m_surfaceData(this),
      m_volumeData(this),
      m_lightData(this)
{}

Group::~Group() = default;

void Group::commit()
{
  m_surfaceData = getParamObject<ObjectArray>("surface");
  m_volumeData = getParamObject<ObjectArray>("volume");
  m_lightData = getParamObject<ObjectArray>("light");
}

void Group::markCommitted()
{
  deviceState()->markSceneChanged();
  Object::markCommitted();
}

TallyGroup::SP Group::makeTallyGroup(TallyModel::SP model, int slot) const
{
  std::vector<TallyGeom::SP> tallyGeometries;
  std::vector<Surface *> surfaces;
  std::vector<TallyVolume::SP> tallyVolumes;
  std::vector<Volume *> volumes;
  std::vector<TallyLight::SP> tallyLights;
  std::vector<Light *> lights;

  // Surfaces //

  if (m_surfaceData) {
    std::for_each(m_surfaceData->handlesBegin(),
        m_surfaceData->handlesEnd(),
        [&](auto *o) {
          auto *s = (Surface *)o;
          if (s && s->isValid())
            surfaces.push_back(s);
        });
  }

  for (auto s : surfaces) {
    TallyGeom::SP geom = s->getTallyGeom(model, slot);
    tallyGeometries.push_back(geom);
  }

  // Volumes //

  if (m_volumeData) {
    std::for_each(
        m_volumeData->handlesBegin(), m_volumeData->handlesEnd(), [&](auto *o) {
          auto *v = (Volume *)o;
          if (v && v->isValid())
            volumes.push_back(v);
        });
  }

  for (auto v : volumes)
    tallyVolumes.push_back(v->getTallyVolume(model, slot));

  // Lights //

  if (m_lightData) {
    std::for_each(
        m_lightData->handlesBegin(), m_lightData->handlesEnd(), [&](auto *o) {
          auto *l = (Light *)o;
          if (l && l->isValid())
            lights.push_back(l);
        });
  }

  for (auto l : lights)
    tallyLights.push_back(l->getTallyLight(model, slot));

  // BNData lightsData = nullptr;
  // if (!tallyLights.empty()) {
  //   lightsData = bnDataCreate(model, slot, BN_OBJECT, tallyLights.size(), tallyLights.data());
  // }

  // Make tally group //

  TallyGroup::SP bg = TallyGroup::create(tallyGeometries,tallyVolumes,tallyLights);
  // TallyGroup::SP bg = bnGroupCreate(model,
  //     slot,
  //     tallyGeometries.data(),
  //     tallyGeometries.size(),
  //     tallyVolumes.data(),
  //     tallyVolumes.size());
  // if (lightsData) {
  //   bnSetData(bg, "lights", lightsData);
  //   bnRelease(lightsData);
  //   bnCommit(bg);
  // }
  // bnGroupBuild(bg);

  // Cleanup //

  // iw - do not release - the anari volumes do track their own handles, don't they!?
 
  // for (auto bnv : tallyVolumes)
  //   bnRelease(bnv);

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

} // namespace tally_device

TALLY_ANARI_TYPEFOR_DEFINITION(tally_device::Group *);
