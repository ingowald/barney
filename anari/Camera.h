// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace barney_device {

struct Camera : public Object
{
  Camera(BarneyGlobalState *s);
  ~Camera() override;

  virtual void commitParameters() override;

  static Camera *createInstance(
      std::string_view type, BarneyGlobalState *state);

  math::float4 imageRegion() const;

  BNCamera barneyCamera() const;

 protected:
  math::float3 m_pos;
  math::float3 m_dir;
  math::float3 m_up;
  math::float4 m_imageRegion;

  BNCamera m_barneyCamera{nullptr};
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct Perspective : public Camera
{
  Perspective(BarneyGlobalState *s);

  void commitParameters() override;
  void finalize() override;

 private:
  float m_fovy{30.f};
  float m_aspect{1.f};
  float m_focusDistance = 0.f;
  float m_apertureRadius = 0.f;
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Camera *, ANARI_CAMERA);
