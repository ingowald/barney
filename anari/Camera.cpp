// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Camera.h"

namespace barney_device {

Camera::Camera(BarneyGlobalState *s) : Object(ANARI_CAMERA, s)
{
  s->objectCounts.cameras++;
}

Camera::~Camera()
{
  deviceState()->objectCounts.cameras--;
}

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
  m_pos = getParam<float3>("position", make_float3(0.f, 0.f, 0.f));
  m_dir = normalize(getParam<float3>("direction", make_float3(0.f, 0.f, 1.f)));
  m_up = normalize(getParam<float3>("up", make_float3(0.f, 1.f, 0.f)));
  m_imageRegion = make_float4(0.f, 0.f, 1.f, 1.f);
  getParam("imageRegion", ANARI_FLOAT32_BOX2, &m_imageRegion);
  markUpdated();
}

const BNCamera *Camera::barneyCamera() const
{
  return &m_barneyCamera;
}

// Subtypes ///////////////////////////////////////////////////////////////////

Perspective::Perspective(BarneyGlobalState *s) : Camera(s) {}

void Perspective::commit()
{
  Camera::commit();

  float fovy = 0.f;
  if (!getParam("fovy", ANARI_FLOAT32, &fovy))
    fovy = anari::radians(60.f);
  float aspect = getParam<float>("aspect", 1.f);

  float3 at =
      make_float3(m_pos.x + m_dir.x, m_pos.y + m_dir.y, m_pos.z + m_dir.z);

  bnPinholeCamera(
      &m_barneyCamera, m_pos, at, m_up, anari::degrees(fovy), aspect);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Camera *);
