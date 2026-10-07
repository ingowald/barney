// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "anari/Splats.h"

namespace BARNEY_NS {
  namespace splats {
    namespace anari {

      Geometry *createSplatsGeometry(BarneyGlobalState *banari)
      {
        return new Splats(banari);
      }

      Splats::Splats(BarneyGlobalState *s)
      : Geometry(s),
        m_index(this),
        m_vertexPosition(this),
        m_vertexRadius(this)
      {}

        /*! return an instance of the corresponding native geometry */
      std::shared_ptr<native::Geometry> Splats::createNative(BarneyGlobalState *s)
      {
        return std::make_shared<native::Splats>(s->getNativeContext(),s->slot);
      }

      void Splats::setBarneyParameters(BNGeom geom)
      {
        native::Splats *native = (native::Splats *)geom;
        native->globalRadius = m_globalRadius;
        native->indexData = {};
        native->vertexPositionData = {};
        native->vertexRadiusData = {};
        if (m_index)
          native->indexData = m_index->asNativePOD();
        if (m_vertexPosition)
          native->vertexPositionData = m_vertexPosition->asNativePOD();
        if (m_vertexRadius)
          native->vertexRadiusData = m_vertexRadius->asNativePOD();
        
        setAttributes(geom);
      }
      
      box3 Splats::bounds() const
      {
        if (!isValid())
          return {};

        box3 result;
        if (m_index) {
          std::for_each(m_index->beginAs<uint32_t>(),
                        m_index->beginAs<uint32_t>() + m_index->totalSize(),
                        [&](uint32_t index) {
                          math::float3 v = *(m_vertexPosition->beginAs<math::float3>() + index);
                          float r = m_vertexRadius ? *(m_vertexRadius->beginAs<float>() + index)
                            : m_globalRadius;
                          result.insert(math::float3{v.x - r, v.y - r, v.z - r});
                          result.insert(math::float3{v.x + r, v.y + r, v.z + r});
                        });
        } else {
          for (size_t i = 0; i < m_vertexPosition->totalSize(); ++i) {
            math::float3 v = *(m_vertexPosition->beginAs<math::float3>() + i);
            float r = m_vertexRadius ? *(m_vertexRadius->beginAs<float>() + i)
              : m_globalRadius;
            result.insert(math::float3{v.x - r, v.y - r, v.z - r});
            result.insert(math::float3{v.x + r, v.y + r, v.z + r});
          }
        }
        return result;
      }
      
      // ==================================================================
      // anari parameter infrastructure
      // ==================================================================
      void Splats::commitParameters()
      {
        Geometry::commitParameters();
        m_index = getParamObject<Array1D>("primitive.index");
        m_vertexPosition = getParamObject<Array1D>("vertex.position");
        m_vertexRadius = getParamObject<Array1D>("vertex.radius");
        m_globalRadius = getParam<float>("radius", 0.01f);
        m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
        m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
        m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
        m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
        m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");
      }
      
      void Splats::finalize()
      {
        if (!m_vertexPosition) {
          reportMessage(ANARI_SEVERITY_WARNING,
                        "missing required parameter 'vertex.position' on splats geometry");
          return;
        }
      }
      
    }
  }
}
