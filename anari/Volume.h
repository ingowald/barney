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

  virtual BNVolume createBarneyVolume(BNModel model, int slot) = 0;
  BNVolume getBarneyVolume(BNModel model, int slot)
  {
    if (!isValid())
      return {};
    if (!isModelTracked(model, slot)) {
      cleanup();
      trackModel(model, slot);
    }
    if (!m_bnVolume) 
      m_bnVolume = createBarneyVolume(model,slot);
    return m_bnVolume;
  }
    
  void cleanup()
  {
    if (m_bnVolume) {
      bnRelease(m_bnVolume);
      m_bnVolume = nullptr;
    }
  }
  

  virtual box3 bounds() const = 0;

  BNVolume m_bnVolume = 0;
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct TransferFunction1D : public Volume
{
  TransferFunction1D(BarneyGlobalState *s);
  void commit() override;
  bool isValid() const override;

  BNVolume createBarneyVolume(BNModel model, int slot) override;

  box3 bounds() const override;

 private:
  helium::IntrusivePtr<SpatialField> m_field;

  box3 m_bounds;

  box1 m_valueRange{0.f, 1.f};
  float m_densityScale{1.f};

  helium::IntrusivePtr<helium::Array1D> m_colorData;
  helium::IntrusivePtr<helium::Array1D> m_opacityData;

  std::vector<math::float4> m_rgbaMap;

};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Volume *, ANARI_VOLUME);
