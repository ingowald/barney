// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#include "Object.h"
// std
#include <atomic>
#include <cstdarg>

namespace barney_device {

// Object definitions /////////////////////////////////////////////////////////

Object::Object(ANARIDataType type, BarneyGlobalState *s)
    : helium::BaseObject(type, s)
{}

void Object::commit()
{
  // no-op
}

bool Object::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  if (name == "valid" && type == ANARI_BOOL) {
    helium::writeToVoidP(ptr, isValid());
    return true;
  }

  return false;
}

bool Object::isValid() const
{
  return true;
}

BarneyGlobalState *Object::deviceState() const
{
  return (BarneyGlobalState *)helium::BaseObject::m_state;
}

bool Object::isModelTracked(BNModel model, int slot) const
{
  return m_bnModel == model && m_slot == slot;
}

void Object::trackModel(BNModel model, int slot)
{
  m_bnModel = model;
  m_slot = slot;
}

BNModel Object::trackedModel() const
{
  return m_bnModel;
}

int Object::trackedSlot() const
{
  return m_slot;
}

// UnknownObject definitions //////////////////////////////////////////////////

UnknownObject::UnknownObject(ANARIDataType type, BarneyGlobalState *s)
    : Object(type, s)
{}

UnknownObject::~UnknownObject() = default;

bool UnknownObject::isValid() const
{
  return false;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Object *);
