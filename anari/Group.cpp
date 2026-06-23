// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
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

  Group::~Group()
  {
    if (m_barneyGroup)
      bnRelease(m_barneyGroup);
  }

  void Group::commitParameters()
  {
    m_surfaceData = getParamObject<ObjectArray>("surface");
    m_volumeData = getParamObject<ObjectArray>("volume");
    m_lightData = getParamObject<ObjectArray>("light");
  }

  void Group::markFinalized()
  {
    const auto lastGroupFinalized = lastFinalized();
    const bool surfaceChanged =
      m_surfaceData.get() != m_lastFinalizedSurfaceData ||
      (m_surfaceData && m_surfaceData->lastFinalized() > lastGroupFinalized);
    const bool volumeChanged =
      m_volumeData.get() != m_lastFinalizedVolumeData ||
      (m_volumeData && m_volumeData->lastFinalized() > lastGroupFinalized);
    const bool lightChanged =
      m_lightData.get() != m_lastFinalizedLightData ||
      (m_lightData && m_lightData->lastFinalized() > lastGroupFinalized);
    const bool structural = surfaceChanged || volumeChanged || lightChanged;

    m_lastFinalizedSurfaceData = m_surfaceData.get();
    m_lastFinalizedVolumeData = m_volumeData.get();
    m_lastFinalizedLightData = m_lightData.get();

    if (structural)
      deviceState()->markStructuralSceneChanged();
    else
      deviceState()->markSceneChanged();
    Object::markFinalized();
  }

  void Group::finalize()
  {}

  
  BNGroup Group::makeBarneyGroup() const
  {
    // Reuse the cached BNGroup unless THIS group's own contents changed.
    // World::fullRebuild() calls makeBarneyGroup() for every instance's group,
    // and markFinalized() flags the scene structural on any group commit, so
    // without this an unrelated change (a light edit, or a cut-plane / overlay
    // instance-transform move that re-commits the world) reconstructs every
    // group's geometry + BLAS — the dominant cost of an overlay drag under MPI.
    // Keyed on the data arrays' lastFinalized() so it also works for the
    // world's zero-instance group (finalized directly).
    const bool ptrChanged = m_surfaceData.get() != m_barneyGroupSurfacePtr
                            || m_volumeData.get() != m_barneyGroupVolumePtr
                            || m_lightData.get() != m_barneyGroupLightPtr;
    helium::TimeStamp contentTS = lastFinalized();
    if (m_surfaceData)
      contentTS = std::max(contentTS, m_surfaceData->lastFinalized());
    if (m_volumeData)
      contentTS = std::max(contentTS, m_volumeData->lastFinalized());
    if (m_lightData)
      contentTS = std::max(contentTS, m_lightData->lastFinalized());
    if (m_barneyGroup && !ptrChanged && m_barneyGroupBuiltAt >= contentTS)
      return m_barneyGroup;
    if (m_barneyGroup) {
      bnRelease(m_barneyGroup);
      m_barneyGroup = 0;
    }

    int slot = deviceState()->slot;
    auto context = deviceState()->tether->context;

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
                      if (s && s->isValid() && s->isVisible())
                        surfaces.push_back(s);
                      else {
                        reportMessage(ANARI_SEVERITY_WARNING,
                                      "encountered invalid surface (%p)",
                                      s);
                      }
                    });
    }

    for (auto s : surfaces) {
      auto bg = s->getBarneyGeom();
      barneyGeometries.push_back(bg);
    }

    // Volumes //

    if (m_volumeData) {
      std::for_each(m_volumeData->handlesBegin(),
                    m_volumeData->handlesEnd(),
                    [&](auto *o) {
                      auto *v = (Volume *)o;
                      if (v && v->isValid() && v->isVisible())
                        volumes.push_back(v);
                      else {
                        reportMessage(ANARI_SEVERITY_WARNING,
                                      "encountered invalid volume (%p)",
                                      v);
                      }
                    });
    }

    for (auto v : volumes)
      barneyVolumes.push_back(v->getBarneyVolume());

    // Lights //

    if (m_lightData) {
      std::for_each(m_lightData->handlesBegin(),
                    m_lightData->handlesEnd(),
                    [&](auto *o) {
                      auto *l = (Light *)o;
                      if (l && l->isValid())
                        lights.push_back(l);
                      else {
                        reportMessage(ANARI_SEVERITY_WARNING,
                                      "encountered invalid light (%p)",
                                      l);
                      }
                    });
    }

    for (auto l : lights) {
      if (l && l->isValid())
        barneyLights.push_back(l->getBarneyLight());
    }

    BNData lightsData = nullptr;
    if (!barneyLights.empty()) {
      lightsData = bnDataCreate(context, slot, BN_OBJECT,
                                barneyLights.size(), barneyLights.data());
    }

    // Make barney group //

    BNGroup bg = bnGroupCreate(context,slot,
                               barneyGeometries.data(),
                               (int)barneyGeometries.size(),
                               barneyVolumes.data(),
                               (int)barneyVolumes.size());
    if (lightsData) {
      bnSetData(bg, "lights", lightsData);
      bnRelease(lightsData);
      bnCommit(bg);
    }
    bnGroupBuild(bg);

    reportMessage(ANARI_SEVERITY_DEBUG,
                  "barney::Group constructed with %zu surfaces, %zu volumes, and %zu lights",
                  barneyGeometries.size(),
                  barneyVolumes.size(),
                  barneyLights.size());

    m_barneyGroup = bg;
    m_barneyGroupBuiltAt = helium::newTimeStamp();
    m_barneyGroupSurfacePtr = m_surfaceData.get();
    m_barneyGroupVolumePtr = m_volumeData.get();
    m_barneyGroupLightPtr = m_lightData.get();
    return m_barneyGroup;
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
