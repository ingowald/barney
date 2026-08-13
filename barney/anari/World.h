// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Array.h"
#include "Instance.h"
#include <array>

namespace BARNEY_NS {
  namespace anari {
    
    struct World : public Object
    {
      World(BarneyGlobalState *s);
      ~World() override;

      bool getProperty(const std::string_view &name,
                       ANARIDataType type,
                       void *ptr,
                       uint64_t size,
                       uint32_t flags) override;

      void commitParameters() override;
      void finalize() override;

      BNModel makeCurrent();
      void markFinalized() override;

    private:
      using InstanceAttributes
      = std::array<std::vector<math::float4>, Instance::Attributes::count>;

      void buildBarneyModel();
      void uploadInstanceAttributes(const InstanceAttributes &attributes);
      void fullRebuild();
      void transformOnlyUpdate();

      helium::ChangeObserverPtr<ObjectArray> m_zeroSurfaceData;
      helium::ChangeObserverPtr<ObjectArray> m_zeroVolumeData;
      helium::ChangeObserverPtr<ObjectArray> m_zeroLightData;
      helium::ChangeObserverPtr<ObjectArray> m_instanceData;

      helium::IntrusivePtr<Group> m_zeroGroup;
      helium::IntrusivePtr<Instance> m_zeroInstance;

      std::vector<Instance *> m_instances;

      TetheredModel::SP tetheredModel;

      BNData m_attributesData[Instance::Attributes::count] = {0,0,0,0,0};
      helium::TimeStamp m_lastBarneyModelBuild{0};
    };

  }
}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::World *, ANARI_WORLD);
