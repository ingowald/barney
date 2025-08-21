// Copyright 2023-2024 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Camera.h"

namespace barney_device {

Camera::Camera(BarneyGlobalState *s) : Object(ANARI_CAMERA, s) {}

Camera::~Camera()
{
  if (m_barneyCamera)
    bnRelease(m_barneyCamera);
}

Camera *Camera::createInstance(std::string_view type, BarneyGlobalState *s)
{
  if (type == "perspective")
    return new Perspective(s);
  else
    return (Camera *)new UnknownObject(ANARI_CAMERA, s);
}

void Camera::commitParameters()
{
  m_pos = getParam<math::float3>("position", math::float3(0.f, 0.f, 0.f));
  if (isnan(m_pos.x+m_pos.y+m_pos.z))
    reportMessage(ANARI_SEVERITY_ERROR, "app set camera.position to NAN coordinates");
    
  m_dir = math::normalize(
      getParam<math::float3>("direction", math::float3(0.f, 0.f, 1.f)));
  if (isnan(m_dir.x+m_dir.y+m_dir.z))
    reportMessage(ANARI_SEVERITY_ERROR, "app set camera.direction to NAN coordinates");
  
  m_up = math::normalize(
      getParam<math::float3>("up", math::float3(0.f, 1.f, 0.f)));
  if (isnan(m_up.x+m_up.y+m_up.z))
    reportMessage(ANARI_SEVERITY_ERROR, "app set camera.up to NAN coordinates");
  m_imageRegion = math::float4(0.f, 0.f, 1.f, 1.f);
  getParam("imageRegion", ANARI_FLOAT32_BOX2, &m_imageRegion);
}

BNCamera Camera::barneyCamera() const
{
  return m_barneyCamera;
}

// Subtypes ///////////////////////////////////////////////////////////////////

Perspective::Perspective(BarneyGlobalState *s) : Camera(s)
{
  assert(deviceState());
  assert(deviceState()->tether);
  assert(deviceState()->tether->context);
  m_barneyCamera = bnCameraCreate(deviceState()->tether->context, "perspective");
}

void Perspective::commitParameters()
{
  Camera::commitParameters();
  m_aspect = getParam<float>("aspect", 1.f);
  m_fovy = getParam<float>("fovy", anari::radians(60.f));
  m_focusDistance = getParam<float>("focusDistance", 0.f);
  m_apertureRadius = getParam<float>("apertureRadius", 0.f);
}

void Perspective::finalize()
{
  bnSet3fc(m_barneyCamera, "up", m_up);
  bnSet3fc(m_barneyCamera, "position", m_pos);
  bnSet3fc(m_barneyCamera, "direction", m_dir);
  bnSet1f(m_barneyCamera, "aspect", m_aspect);
  bnSet1f(m_barneyCamera, "focusDistance", m_focusDistance);
  bnSet1f(m_barneyCamera, "apertureRadius", m_apertureRadius);
  bnSet1f(m_barneyCamera, "fovy", anari::degrees(m_fovy));
  bnCommit(m_barneyCamera);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Camera *);
