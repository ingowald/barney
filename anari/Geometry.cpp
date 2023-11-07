// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Geometry.h"
// std
#include <numeric>

namespace barney_device {

Geometry::Geometry(BarneyGlobalState *s) : Object(ANARI_GEOMETRY, s)
{
  s->objectCounts.groups++;
}

Geometry::~Geometry()
{
  deviceState()->objectCounts.groups--;
}

Geometry *Geometry::createInstance(
    std::string_view subtype, BarneyGlobalState *s)
{
  if (subtype == "triangle")
    return new Triangle(s);
  else
    return (Geometry *)new UnknownObject(ANARI_GEOMETRY, s);
}

// Subtypes ///////////////////////////////////////////////////////////////////

Triangle::Triangle(BarneyGlobalState *s) : Geometry(s) {}

void Triangle::commit()
{
  Geometry::commit();

  cleanup();

  m_index = getParamObject<Array1D>("primitive.index");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on triangle geometry");
    return;
  }

  m_generatedIndices.clear();

  m_vertexPosition->addCommitObserver(this);
  if (m_index) {
    m_index->addCommitObserver(this);
  } else {
    m_generatedIndices.resize(m_vertexPosition->totalSize());
    std::iota(m_generatedIndices.begin(), m_generatedIndices.end(), 0);
  }
}

BNGeom Triangle::makeBarneyGeometry(
    BNDataGroup dg, const BNMaterial *material) const
{
  auto ctx = deviceState()->context;
  return bnTriangleMeshCreate(dg,
      material,
      m_index ? (const int3 *)m_index->data()
              : (const int3 *)m_generatedIndices.data(),
      m_index ? m_index->size() : (m_generatedIndices.size() / 3),
      m_vertexPosition->dataAs<float3>(),
      m_vertexPosition->totalSize(),
      nullptr,
      nullptr);
}

void Triangle::cleanup()
{
  if (m_index)
    m_index->removeCommitObserver(this);
  if (m_vertexPosition)
    m_vertexPosition->removeCommitObserver(this);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Geometry *);
