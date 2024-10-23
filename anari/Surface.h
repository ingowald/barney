// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Geometry.h"
#include "Material.h"

namespace tally_device {

struct Surface : public Object
{
  Surface(TallyGlobalState *s);
  ~Surface() override;

  void commit() override;
  void markCommitted() override;

  uint32_t id() const;
  const Geometry *geometry() const;
  const Material *material() const;

  TallyGeom::SP getTallyGeom(TallyModel::SP model, int slot);

  bool isValid() const override;

 private:
  void setTallyParameters();
  void cleanup();

  uint32_t m_id{~0u};
  helium::IntrusivePtr<Geometry> m_geometry;
  helium::IntrusivePtr<Material> m_material;

  TallyGeom::SP m_bnGeom{nullptr};
  TallyMaterial::SP m_bnMat{nullptr};
};

} // namespace tally_device

TALLY_ANARI_TYPEFOR_SPECIALIZATION(tally_device::Surface *, ANARI_SURFACE);
