// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "BarneyGlobalState.h"
// helium
#include "helium/BaseObject.h"
// std
#include <string_view>

namespace barney_device {

struct Object : public helium::BaseObject
{
  Object(ANARIDataType type, BarneyGlobalState *s);
  virtual ~Object() = default;

  virtual bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags);

  virtual void commit();

  bool isValid() const override;

  BarneyGlobalState *deviceState() const;
};

struct UnknownObject : public Object
{
  UnknownObject(ANARIDataType type, BarneyGlobalState *s);
  ~UnknownObject() override;
  bool isValid() const override;
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Object *, ANARI_OBJECT);
