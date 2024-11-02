// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Instance.h"
// anari
#include <anari/anari_cpp/ext/linalg.h>
// CUDA
// #include <vector_functions.h>

namespace barney_device {

Instance::Instance(BarneyGlobalState *s) : Object(ANARI_INSTANCE, s) {}

Instance::~Instance() = default;

void Instance::commit()
{
  math::mat4 xfm = anari::math::identity;
  getParam("transform", ANARI_FLOAT32_MAT4, &xfm);
  (anari::math::float3&)m_xfm.xfm.l.vx
    = anari::math::float3(xfm[0].x, xfm[0].y, xfm[0].z);
  (anari::math::float3&)m_xfm.xfm.l.vy
    = anari::math::float3(xfm[1].x, xfm[1].y, xfm[1].z);
  (anari::math::float3&)m_xfm.xfm.l.vz
    = anari::math::float3(xfm[2].x, xfm[2].y, xfm[2].z);
  (anari::math::float3&)m_xfm.xfm.p
    = anari::math::float3(xfm[3].x, xfm[3].y, xfm[3].z);

  m_group = getParamObject<Group>("group");
  if (!m_group)
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'group' on ANARIInstance");
}

void Instance::markCommitted()
{
  deviceState()->markSceneChanged();
  Object::markCommitted();
}

bool Instance::isValid() const
{
  return m_group;
}

const Group *Instance::group() const
{
  return m_group.ptr;
}

const BNTransform *Instance::barneyTransform() const
{
  return &m_xfm;
}

box3 Instance::bounds() const
{
  math::mat4 xfm;
  xfm[0] = math::float4(m_xfm.xfm.l.vx.x, m_xfm.xfm.l.vx.y, m_xfm.xfm.l.vx.z, 0.f);
  xfm[1] = math::float4(m_xfm.xfm.l.vy.x, m_xfm.xfm.l.vy.y, m_xfm.xfm.l.vy.z, 0.f);
  xfm[2] = math::float4(m_xfm.xfm.l.vz.x, m_xfm.xfm.l.vz.y, m_xfm.xfm.l.vz.z, 0.f);
  xfm[3] = math::float4(m_xfm.xfm.p.x, m_xfm.xfm.p.y, m_xfm.xfm.p.z, 1.f);

  box3 result = group()->bounds();

  math::float4 lower(result.lower, 1.f);
  math::float4 upper(result.upper, 1.f);

  lower = mul(xfm, lower);
  upper = mul(xfm, upper);

  result.lower = math::float3(lower.x, lower.y, lower.z);
  result.upper = math::float3(upper.x, upper.y, upper.z);

  return result;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Instance *);
