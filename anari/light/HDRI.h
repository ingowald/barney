// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "light/Light.h"
#include "helium/array/Array1D.h"
#include "helium/array/Array2D.h"

namespace barney_device {

  typedef helium::IntrusivePtr<helium::Array2D> Array2DPtr;
  struct HDRILight : public Light
  {
    HDRILight(BarneyGlobalState *s);
    ~HDRILight() override;

    void commitParameters() override;
    void finalize() override;

  private:
    const char *bnSubtype() const override;
    void setBarneyParameters() override;
    
    float         m_scale     { 1.f };
    math::float3  m_up        { 0.f, 0.f, 1.f };
    math::float3  m_direction { 1.f, 0.f, 0.f };
    Array2DPtr    m_radiance;
  };

} // ::barney_device

