// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "BarneyGlobalState.h"
#include "common.h"
// helium
#include "helium/BaseObject.h"
#include "helium/utility/ChangeObserverPtr.h"
// std
#include <string_view>

namespace barney_device {

struct Object : public helium::BaseObject
{
  Object(ANARIDataType type, BarneyGlobalState *s);
  virtual ~Object() = default;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

  // BNContext getContext() const;
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
