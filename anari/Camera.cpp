// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Camera.h"

namespace tally_device {

Camera::Camera(TallyGlobalState *s) : Object(ANARI_CAMERA, s)
{}

Camera::~Camera() = default;

Camera *Camera::createInstance(std::string_view type, TallyGlobalState *s)
{
  if (type == "perspective")
    return new Perspective(s);
  else
    return (Camera *)new UnknownObject(ANARI_CAMERA, s);
}

void Camera::commit()
{
  if (!m_tallyCamera)
    m_tallyCamera = TallyCamera::create("perspective");
  m_pos = getParam<math::float3>("position", math::float3(0.f, 0.f, 0.f));
  m_dir = math::normalize(
      getParam<math::float3>("direction", math::float3(0.f, 0.f, 1.f)));
  m_up = math::normalize(
      getParam<math::float3>("up", math::float3(0.f, 1.f, 0.f)));
  m_imageRegion = math::float4(0.f, 0.f, 1.f, 1.f);
  getParam("imageRegion", ANARI_FLOAT32_BOX2, &m_imageRegion);
  markUpdated();
}

  TallyCamera::SP Camera::tallyCamera() const
{
  return m_tallyCamera;
}

// Subtypes ///////////////////////////////////////////////////////////////////

Perspective::Perspective(TallyGlobalState *s) : Camera(s) {}

void Perspective::commit()
{
  Camera::commit();
#if TALLY
  bnSet3fc(m_tallyCamera,"up",(const float3&)m_up);
  bnSet3fc(m_tallyCamera,"position", (const float3&)m_pos);
  bnSet3fc(m_tallyCamera,"direction",(const float3&)m_dir);
  float aspect = getParam<float>("aspect", 1.f);
  bnSet1f(m_tallyCamera,"aspect",aspect);
  float fovy = getParam<float>("fovy", anari::radians(60.f));
  bnSet1f(m_tallyCamera,"fovy",anari::degrees(fovy));
  bnCommit(m_tallyCamera);
#endif
}

} // namespace tally_device

TALLY_ANARI_TYPEFOR_DEFINITION(tally_device::Camera *);
