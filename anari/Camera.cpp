// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Camera.h"

namespace barney_device {

Camera::Camera(BarneyGlobalState *s) : Object(ANARI_CAMERA, s)
{}

Camera::~Camera() = default;

Camera *Camera::createInstance(std::string_view type, BarneyGlobalState *s)
{
  if (type == "perspective")
    return new Perspective(s);
#if 0
  else if (type == "orthographic")
    return new Orthographic(s);
#endif
  else
    return (Camera *)new UnknownObject(ANARI_CAMERA, s);
}

void Camera::commit()
{
  if (!m_barneyCamera)
    m_barneyCamera = bnCameraCreate(deviceState()->context,"perspective");
  m_pos = getParam<math::float3>("position", math::float3(0.f, 0.f, 0.f));
  m_dir = math::normalize(
      getParam<math::float3>("direction", math::float3(0.f, 0.f, 1.f)));
  m_up = math::normalize(
      getParam<math::float3>("up", math::float3(0.f, 1.f, 0.f)));
  m_imageRegion = math::float4(0.f, 0.f, 1.f, 1.f);
  getParam("imageRegion", ANARI_FLOAT32_BOX2, &m_imageRegion);
  markUpdated();
}

const BNCamera Camera::barneyCamera() const
{
  return m_barneyCamera;
}

// Subtypes ///////////////////////////////////////////////////////////////////

Perspective::Perspective(BarneyGlobalState *s) : Camera(s) {}

void Perspective::commit()
{
  Camera::commit();

  float fovy = 0.f;
  if (!getParam("fovy", ANARI_FLOAT32, &fovy))
    fovy = 60.f;//anari::radians(60.f);
  bnSet3fc(m_barneyCamera,"up",(const float3&)m_up);
  bnSet3fc(m_barneyCamera,"position", (const float3&)m_pos);
  bnSet3fc(m_barneyCamera,"direction",(const float3&)m_dir);
  bnSet1f(m_barneyCamera,"aspect",getParam<float>("aspect", 1.f));
  bnSet1f(m_barneyCamera,"fovy",anari::degrees(fovy));
  bnCommit(m_barneyCamera);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Camera *);
