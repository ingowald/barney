// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Array.h"
#include "Object.h"
#include "SpatialField.h"

namespace tally_device {

struct Volume : public Object
{
  Volume(TallyGlobalState *s);
  ~Volume() override;

  static Volume *createInstance(std::string_view subtype, TallyGlobalState *s);

  void markCommitted() override;

  TallyVolume::SP getTallyVolume(TallyModel::SP model, int slot);

  virtual box3 bounds() const = 0;

 protected:
  virtual TallyVolume::SP createTallyVolume(TallyModel::SP model, int slot) = 0;
  virtual void setTallyParameters() = 0;
  void cleanup();

  TallyVolume::SP m_bnVolume{nullptr};
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct TransferFunction1D : public Volume
{
  TransferFunction1D(TallyGlobalState *s);
  void commit() override;
  bool isValid() const override;

  TallyVolume::SP createTallyVolume(TallyModel::SP model, int slot) override;

  box3 bounds() const override;

 private:
  void setTallyParameters() override;

  helium::IntrusivePtr<SpatialField> m_field;

  box3 m_bounds;

  box1 m_valueRange{0.f, 1.f};
  float m_densityScale{1.f};

  helium::IntrusivePtr<helium::Array1D> m_colorData;
  helium::IntrusivePtr<helium::Array1D> m_opacityData;
  bool needsOpacityData;

  std::vector<math::float4> m_rgbaMap;
};

} // namespace tally_device

TALLY_ANARI_TYPEFOR_SPECIALIZATION(tally_device::Volume *, ANARI_VOLUME);
