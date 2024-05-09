// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Group.h"
#include <iostream>

namespace barney_device {

Group::Group(BarneyGlobalState *s)
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

BNGroup Group::makeBarneyGroup(BNModel model, int slot) const
{
  std::vector<BNGeom> barneyGeometries;
  std::vector<Surface *> surfaces;
  std::vector<BNVolume> barneyVolumes;
  std::vector<Volume *> volumes;
  std::vector<BNLight> barneyLights;
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

  for (auto s : surfaces)  {
    barneyGeometries.push_back(s->getBarneyGeom(model, slot));
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
    barneyVolumes.push_back(v->makeBarneyVolume(model, slot));

  // Lights //

  if (m_lightData) {
    std::for_each(
        m_lightData->handlesBegin(), m_lightData->handlesEnd(), [&](auto *o) {
          auto *l = (Light *)o;
          if (l && l->isValid())
            lights.push_back(l);
        });
  }

  for (auto l : lights) {
    auto bnl = bnLightCreate(model, slot, "directional");
    l->setBarneyParameters(bnl);
    barneyLights.push_back(bnl);
  }

  BNData lightsData = nullptr;
  if (!barneyLights.empty()) {
    lightsData = bnDataCreate(
        model, slot, BN_OBJECT, barneyLights.size(), barneyLights.data());
  }

  // Make barney group //

  BNGroup bg = bnGroupCreate(model,
      slot,
      barneyGeometries.data(),
      barneyGeometries.size(),
      barneyVolumes.data(),
      barneyVolumes.size());
  if (lightsData) {
    bnSetData(bg, "lights", lightsData);
    bnRelease(lightsData);
    bnCommit(bg);
  }
  bnGroupBuild(bg);

  // Cleanup //

  for (auto bnv : barneyVolumes)
    bnRelease(bnv);

  for (auto bnl : barneyLights)
    bnRelease(bnl);

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

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Group *);
