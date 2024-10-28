// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
#include "common.h"
#include "helium/array/Array1D.h"
#include "helium/array/Array2D.h"
#include "common.h"
// CUDA
// #include <vector_functions.h>
// #include <iostream>

namespace barney_device {

  struct Light : public Object
  {
    Light(BarneyGlobalState *s);
    ~Light() override;

    static Light *createInstance(std::string_view type, BarneyGlobalState *state);

    void markCommitted() override;
    virtual void commit() override;

    BNLight getBarneyLight(BNModel model, int slot);

  protected:
    virtual const char *bnSubtype() const = 0;
    virtual void setBarneyParameters() = 0;

    void cleanup();

    math::float3 m_color{1.f, 1.f, 1.f};

    BNLight m_bnLight{nullptr};
  };

  // Subtypes ///////////////////////////////////////////////////////////////////

  struct Directional : public Light
  {
    Directional(BarneyGlobalState *s);

    void commit() override;

  private:
    const char *bnSubtype() const;
    void setBarneyParameters() override;

    /*! SPEC: main emission direction of the directional light */
    math::float3 m_direction{0.f, 0.f, -1.f};
    
    /*! SPEC: the amount of light arriving at a surface point,
        assuming the light is oriented towards to the surface, in
        W/m2 */
    float m_irradiance = NAN;
    /*! the amount of light emitted in a direction, in W/sr/m2;
        irradiance takes precedence if also specified */
    float m_radiance = 1.f;
  };

  struct PointLight : public Light
  {
    PointLight(BarneyGlobalState *s);

    void commit() override;

  private:
    const char *bnSubtype() const;
    void setBarneyParameters() override;

    math::float3 m_position{0.f, 0.f, 0.f};
    
    /*! SPEC: the overall amount of light emitted by the light in a
        direction, in W/sr */
    float        m_intensity = NAN;

    /*! SPEC: the overall amount of light energy emitted, in W;
        intensity takes precedence if also specified */
    float        m_power = 1.f;
  };

  struct HDRILight : public Light
  {
    HDRILight(BarneyGlobalState *s);

    void commit() override;

  private:
    const char *bnSubtype() const;
    void setBarneyParameters() override;

    math::float3 m_up{0.f, 0.f, 1.f};
    math::float3 m_direction{1.f, 0.f, 0.f};
    helium::IntrusivePtr<helium::Array2D> m_radiance;
  };
  
} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Light *, ANARI_LIGHT);
