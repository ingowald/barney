// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Array.h"
#include "Object.h"
#include "SpatialField.h"

namespace barney_device {

struct Volume : public Object
{
  Volume(BarneyGlobalState *s);
  ~Volume() override;

  static Volume *createInstance(std::string_view subtype, BarneyGlobalState *s);

  void markCommitted() override;

  virtual BNVolume makeBarneyVolume(BNDataGroup dg) const = 0;

  virtual anari::box3 bounds() const = 0;
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct TransferFunction1D : public Volume
{
  TransferFunction1D(BarneyGlobalState *s);
  void commit() override;

  BNVolume makeBarneyVolume(BNDataGroup dg) const override;

  anari::box3 bounds() const override;

 private:
  void cleanup();

  helium::IntrusivePtr<SpatialField> m_field;

  anari::box3 m_bounds;

  anari::box1 m_valueRange{0.f, 1.f};
  float m_densityScale{1.f};

  helium::IntrusivePtr<helium::Array1D> m_colorData;
  helium::IntrusivePtr<helium::Array1D> m_opacityData;

  std::vector<float4> m_rgbaMap;
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Volume *, ANARI_VOLUME);