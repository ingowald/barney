// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "BarneyGlobalState.h"
#include "common.h"
// helium
#include "helium/BaseObject.h"
#include "helium/utility/ChangeObserverPtr.h"
// std
#include <string_view>

namespace BANARI_NS {
    
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

    BarneyGlobalState *deviceState() const;
  };

  struct UnknownObject : public Object
  {
    UnknownObject(ANARIDataType type,
                  std::string_view subtype,
                  BarneyGlobalState *s);
    ~UnknownObject() override;
    bool isValid() const override;
  };

}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BANARI_NS::Object *, ANARI_OBJECT);
