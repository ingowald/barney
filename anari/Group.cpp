// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Group.h"
#include <iostream>
#include "common.h"

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

BNGroup Group::makeBarneyGroup(BNContext context// , int slot
                               ) const
{
  std::vector<BNGeom> barneyGeometries;
  std::vector<Surface *> surfaces;
  std::vector<BNVolume> barneyVolumes;
  std::vector<Volume *> volumes;
  std::vector<BNLight> barneyLights;
  std::vector<Light *> lights;

  int slot = 0;
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
    BNGeom geom = s->getBarneyGeom(context// , slot
                                   );
    barneyGeometries.push_back(geom);
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

  for (auto v : volumes) {
    barneyVolumes.push_back(v->getBarneyVolume(context// , slot
                                               ));
  }

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
    barneyLights.push_back(l->getBarneyLight(context// , slot
                                             ));

  BNData lightsData = nullptr;
  if (!barneyLights.empty()) {
    lightsData = bnDataCreate
      (context, slot, BN_OBJECT, barneyLights.size(), barneyLights.data());
  }

  // Make barney group //
  
  BNGroup bg = bnGroupCreate(context,
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

  // iw - do not release - the anari volumes do track their own handles, don't they!?
 
  // for (auto bnv : barneyVolumes)
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

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Group *);
