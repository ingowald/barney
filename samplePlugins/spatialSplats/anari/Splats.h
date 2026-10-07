// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "barney/native/Context.h"
#include "barney/anari/Device.h"
#include "barney/anari/Geometry.h"

// ours:
#include "native/Splats.h"

namespace BARNEY_NS {
  namespace splats {
    namespace anari {
      using namespace BARNEY_NS::anari;
      
      struct Splats : public Geometry
      {
        // the static creator fct we're exporting to the barney plugin
        // infrastructure
        static anari::SpatialField *create(BarneyGlobalState *banari);
        
        Splats(BarneyGlobalState *s);

        // ==================================================================
        // anari::Geometry functionality
        // ==================================================================
        
        /*! return an instance of the corresponding native geometry */
        std::shared_ptr<native::Geometry> createNative(BarneyGlobalState *s) override;

        void setBarneyParameters(BNGeom geom) override;
        box3 bounds() const override;
        
        // ==================================================================
        // anari parameter infrastructure
        // ==================================================================
        void commitParameters() override;
        void finalize() override;

        /*! @{ some sample data fields; we use the same as a
            anari::sphere geometry so we can use existing anari model
            viewer to populate this from spheres data */
        helium::ChangeObserverPtr<Array1D> m_index;
        helium::ChangeObserverPtr<Array1D> m_vertexPosition;
        helium::ChangeObserverPtr<Array1D> m_vertexRadius;
        float m_globalRadius{0.f};
        /*! @} */

        // native::PODData::SP m_native_vertexPosition;
        // native::PODData::SP m_native_vertexRadius;
        
        /* note we do NOT store a shared-ptr to the native one,
           because in theory there could multiple native::Splats for each
           anari::Splats if the latter is used in mulitple anari::surfaces.*/
      };
      
      // the static creator fct we're exporting to the barney plugin
      // infrastructure
      Geometry *createSplatsGeometry(anari::BarneyGlobalState *banari);
        
    }
  }
}
