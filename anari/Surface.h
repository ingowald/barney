// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Geometry.h"
#include "Material.h"

namespace barney_device {

struct Surface : public Object
{
  Surface(BarneyGlobalState *s);
  ~Surface() override;

  void commit() override;
  void markCommitted() override;

  uint32_t id() const;
  const Geometry *geometry() const;
  const Material *material() const;

  BNGeom getBarneyGeom(BNModel model, int slot);

  bool isValid() const override;

 private:
  void setBarneyParameters();
  void cleanup();

  uint32_t m_id{~0u};
  helium::IntrusivePtr<Geometry> m_geometry;
  helium::IntrusivePtr<Material> m_material;

  BNGeom m_bnGeom{nullptr};
  BNMaterial m_bnMat{nullptr};
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Surface *, ANARI_SURFACE);
