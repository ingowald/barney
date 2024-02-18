// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace barney_device {

struct Material : public Object
{
  Material(BarneyGlobalState *s);
  ~Material() override;

  static Material *createInstance(
      std::string_view subtype, BarneyGlobalState *s);

  void markCommitted() override;

  const BNMaterialHelper *barneyMaterial() const;

 protected:
  BNMaterial m_bnMaterial;
  BNMaterialHelper m_bnMaterialData;
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct Matte : public Material
{
  Matte(BarneyGlobalState *s);
  void commit() override;
};

struct PhysicallyBased : public Material
{
  PhysicallyBased(BarneyGlobalState *s);
  void commit() override;
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Material *, ANARI_MATERIAL);
