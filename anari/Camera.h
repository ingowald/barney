// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace tally_device {

struct Camera : public Object
{
  Camera(TallyGlobalState *s);
  ~Camera() override;

  virtual void commit() override;

  static Camera *createInstance(
      std::string_view type, TallyGlobalState *state);

  math::float4 imageRegion() const;

  TallyCamera::SP tallyCamera() const;

 protected:
  math::float3 m_pos;
  math::float3 m_dir;
  math::float3 m_up;
  math::float4 m_imageRegion;

  TallyCamera::SP m_tallyCamera = 0;
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct Perspective : public Camera
{
  Perspective(TallyGlobalState *s);

  void commit() override;

 private:
  float m_fovy{30.f};
  float m_aspect{1.f};
};

} // namespace tally_device

TALLY_ANARI_TYPEFOR_SPECIALIZATION(tally_device::Camera *, ANARI_CAMERA);
