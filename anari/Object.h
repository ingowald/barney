// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include "TallyGlobalState.h"
// helium
#include "helium/BaseObject.h"
#include "helium/utility/ChangeObserverPtr.h"
// std
#include <string_view>

namespace tally_device {

struct Object : public helium::BaseObject
{
  Object(ANARIDataType type, TallyGlobalState *s);
  virtual ~Object() = default;

  virtual bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags);

  virtual void commit();

  bool isValid() const override;

  TallyGlobalState *deviceState() const;

 protected:
  // Return if this object is tracking the currently used model
  bool isModelTracked(TallyModel::SP model, int slot = 0) const;
  void trackModel(TallyModel::SP model, int slot = 0);

  TallyModel::SP trackedModel() const;
  int trackedSlot() const;

 private:
  TallyModel::SP m_bnModel; // not an owning reference
  int m_slot{-1};
};

struct UnknownObject : public Object
{
  UnknownObject(ANARIDataType type, TallyGlobalState *s);
  ~UnknownObject() override;
  bool isValid() const override;
};

} // namespace tally_device

TALLY_ANARI_TYPEFOR_SPECIALIZATION(tally_device::Object *, ANARI_OBJECT);
