// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Instance.h"
// anari
#include <anari/anari_cpp/ext/linalg.h>

namespace barney_device {

Instance::Instance(BarneyGlobalState *s) : Object(ANARI_INSTANCE, s)
{
  s->objectCounts.instances++;
}

Instance::~Instance()
{
  deviceState()->objectCounts.instances--;
}

void Instance::commit()
{
  anari::math::mat4 xfm = anari::math::identity;
  getParam("transform", ANARI_FLOAT32_MAT4, &xfm);
  m_xfm.xfm.l.vx = make_float3(xfm[0].x, xfm[0].y, xfm[0].z);
  m_xfm.xfm.l.vy = make_float3(xfm[1].x, xfm[1].y, xfm[1].z);
  m_xfm.xfm.l.vz = make_float3(xfm[2].x, xfm[2].y, xfm[2].z);
  m_xfm.xfm.p = make_float3(xfm[3].x, xfm[3].y, xfm[3].z);

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

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Instance *);
