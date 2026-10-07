// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/anari/Geometry.h"
#include "barney/anari/Material.h"

namespace BARNEY_NS {
  namespace anari {

    struct Surface : public Object
    {
      Surface(BarneyGlobalState *s);
      ~Surface() override;

      void commitParameters() override;
      void finalize() override;
      void markFinalized() override;

      uint32_t id() const;
      const Geometry *geometry() const;
      const Material *material() const;

      /* iw - note this function still returns old native-api 'BNGeom'
         type even though below e alrady use
         native::Geometry::SP. BNGeom is simply a pointer to that
         native geom, plus some internal refcounting on the native
         layer, which we don't need becuase we use the shared_ptr
         directly. */
      BNGeom getBarneyGeom();

      bool isValid() const override;
      bool isVisible() const;

    private:
      void setBarneyParameters();
      void cleanup();

      uint32_t m_id{~0u};
      bool m_visible{true};
      helium::ChangeObserverPtr<Geometry> m_geometry;
      helium::ChangeObserverPtr<Material> m_material;

      
      BNMaterial m_bnMat{nullptr};
      std::shared_ptr<native::Geometry> m_nativeGeom;
    };

  }
}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::Surface *, ANARI_SURFACE);
