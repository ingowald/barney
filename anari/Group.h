// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "light/Light.h"
#include "Surface.h"
#include "Volume.h"
// std
#include <vector>

namespace barney_device {

  struct Group : public Object
  {
    Group(BarneyGlobalState *s);
    ~Group() override;

    void commitParameters() override;
    void finalize() override;
    void markFinalized() override;

    BNGroup makeBarneyGroup() const;

    box3 bounds() const;

  private:
    helium::ChangeObserverPtr<ObjectArray> m_surfaceData;
    helium::ChangeObserverPtr<ObjectArray> m_volumeData;
    helium::ChangeObserverPtr<ObjectArray> m_lightData;

    ObjectArray *m_lastFinalizedSurfaceData{nullptr};
    ObjectArray *m_lastFinalizedVolumeData{nullptr};
    ObjectArray *m_lastFinalizedLightData{nullptr};

    // Cached barney group + the state it was built from. makeBarneyGroup()
    // rebuilds (re-uploads geometry, rebuilds the BLAS) only when this group's
    // own contents change — NOT on every World::fullRebuild(). markFinalized()
    // marks the whole scene structural on any group commit, so without this an
    // unrelated change (a light edit, or a cut-plane/overlay instance-transform
    // move that re-commits the world) reconstructs every group's geometry.
    // Keyed on the data arrays' lastFinalized() so it also works for the
    // world's zero-instance group (finalized directly, not via markFinalized()).
    mutable BNGroup m_barneyGroup{0};
    mutable helium::TimeStamp m_barneyGroupBuiltAt{0};
    mutable ObjectArray *m_barneyGroupSurfacePtr{nullptr};
    mutable ObjectArray *m_barneyGroupVolumePtr{nullptr};
    mutable ObjectArray *m_barneyGroupLightPtr{nullptr};
  };

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Group *, ANARI_GROUP);
