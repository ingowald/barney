// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

// std
#include <cfloat>
/// anari
#include "helium/array/Array3D.h"
// ours
#include "SpatialField.h"

namespace barney_device {

SpatialField::SpatialField(BarneyGlobalState *s)
    : Object(ANARI_SPATIAL_FIELD, s)
{
  s->objectCounts.spatialFields++;
}

SpatialField::~SpatialField()
{
  deviceState()->objectCounts.spatialFields--;
}

SpatialField *SpatialField::createInstance(
    std::string_view subtype, BarneyGlobalState *s)
{
  if (subtype == "unstructured")
    return new UnstructuredField(s);
  else
    return (SpatialField *)new UnknownObject(ANARI_SPATIAL_FIELD, s);
}

void SpatialField::markCommitted()
{
  deviceState()->markSceneChanged();
  Object::markCommitted();
}

// Subtypes ///////////////////////////////////////////////////////////////////

UnstructuredField::UnstructuredField(BarneyGlobalState *s) : SpatialField(s) {}

void UnstructuredField::commit()
{
  Object::commit();

  m_params.vertexPosition = getParamObject<helium::Array1D>("vertex.position");
  m_params.vertexData = getParamObject<helium::Array1D>("vertex.data");
  m_params.index = getParamObject<helium::Array1D>("index");
  m_params.cellIndex = getParamObject<helium::Array1D>("cell.index");

  if (!m_params.vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on unstructured spatial field");
    return;
  }

  if (!m_params.vertexData) { // currently vertex data only!
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.data' on unstructured spatial field");
    return;
  }

  if (!m_params.index) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'index' on unstructured spatial field");
    return;
  }

  if (!m_params.cellIndex) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'cell.index' on unstructured spatial field");
    return;
  }

  m_params.gridData = getParamObject<helium::ObjectArray>("grid.data");
  m_params.gridDomains = getParamObject<helium::Array1D>("grid.domains");

  auto *vertexPosition = m_params.vertexPosition->beginAs<float3>();
  auto *vertexData = m_params.vertexData->beginAs<float>();
  auto *index = m_params.index->beginAs<uint64_t>();
  auto *cellIndex = m_params.cellIndex->beginAs<uint64_t>();

  size_t numVerts = m_params.vertexPosition->size();
  size_t numCells = m_params.cellIndex->size();
  size_t numIndices = m_params.index->endAs<uint64_t>() - index;

  m_generatedVertices.clear();
  m_generatedTets.clear();
  m_generatedPyrs.clear();
  m_generatedWedges.clear();
  m_generatedHexes.clear();

  m_bounds = {
      float3{FLT_MAX, FLT_MAX, FLT_MAX}, float3{-FLT_MAX, -FLT_MAX, -FLT_MAX}};

  for (size_t i = 0; i < numIndices; ++i) {
    m_bounds.insert(vertexPosition[index[i]]);
  }

  for (size_t i = 0; i < numVerts; ++i) {
    float3 pos = vertexPosition[i];
    float value = vertexData[i];
    m_generatedVertices.push_back(pos.x);
    m_generatedVertices.push_back(pos.y);
    m_generatedVertices.push_back(pos.z);
    m_generatedVertices.push_back(value);
  }

  for (size_t i = 0; i < numCells; ++i) {
    uint64_t firstIndex = cellIndex[i];
    uint64_t lastIndex = i < numCells - 1 ? cellIndex[i + 1] : numIndices;

    if (lastIndex - firstIndex == 4) {
      for (uint64_t j = firstIndex; j < lastIndex; ++j) {
        m_generatedTets.push_back(index[j]);
      }
    } else if (lastIndex - firstIndex == 5) {
      for (uint64_t j = firstIndex; j < lastIndex; ++j) {
        m_generatedPyrs.push_back(index[j]);
      }
    } else if (lastIndex - firstIndex == 6) {
      for (uint64_t j = firstIndex; j < lastIndex; ++j) {
        m_generatedWedges.push_back(index[j]);
      }
    } else if (lastIndex - firstIndex == 8) {
      for (uint64_t j = firstIndex; j < lastIndex; ++j) {
        m_generatedHexes.push_back(index[j]);
      }
    }
  }

  if (m_params.gridData && m_params.gridDomains) {
    m_generatedGridOffsets.clear();
    m_generatedGridDims.clear();
    m_generatedGridDomains.clear();
    m_generatedGridScalars.clear();

    size_t numGrids = m_params.gridData->totalSize();
    auto *gridData = (helium::Array3D **)m_params.gridData->handlesBegin();
    auto *gridDomains = m_params.gridDomains->beginAs<anari::box3>();

    for (size_t i = 0; i < numGrids; ++i) {
      const helium::Array3D *gd = *(gridData + i);
      const anari::box3 domain = *(gridDomains + i);

      m_generatedGridOffsets.push_back(m_generatedGridScalars.size());

      // from anari's array3d we get the number of vertices, not cells!
      m_generatedGridDims.push_back(gd->size().x - 1);
      m_generatedGridDims.push_back(gd->size().y - 1);
      m_generatedGridDims.push_back(gd->size().z - 1);

      anari::box1 valueRange{FLT_MAX, -FLT_MAX};
      for (unsigned z = 0; z < gd->size().z; ++z)
        for (unsigned y = 0; y < gd->size().y; ++y)
          for (unsigned x = 0; x < gd->size().x; ++x) {
            size_t index =
                z * size_t(gd->size().x) * gd->size().y + y * gd->size().x + x;
            float f = gd->dataAs<float>()[index];
            m_generatedGridScalars.push_back(f);
            valueRange.insert(f);
          }

      m_generatedGridDomains.push_back(domain.lower.x);
      m_generatedGridDomains.push_back(domain.lower.y);
      m_generatedGridDomains.push_back(domain.lower.z);
      m_generatedGridDomains.push_back(valueRange.lower);
      m_generatedGridDomains.push_back(domain.upper.x);
      m_generatedGridDomains.push_back(domain.upper.y);
      m_generatedGridDomains.push_back(domain.upper.z);
      m_generatedGridDomains.push_back(valueRange.upper);
    }
  }
}

BNScalarField UnstructuredField::makeBarneyScalarField(BNDataGroup dg) const
{
  auto ctx = deviceState()->context;
  return bnUMeshCreate(dg,
      m_generatedVertices.data(),
      m_generatedVertices.size() / 4,
      m_generatedTets.data(),
      m_generatedTets.size() / 4,
      m_generatedPyrs.data(),
      m_generatedPyrs.size() / 5,
      m_generatedWedges.data(),
      m_generatedWedges.size() / 6,
      m_generatedHexes.data(),
      m_generatedHexes.size() / 8,
      m_generatedGridOffsets.size(),
      m_generatedGridOffsets.data(),
      m_generatedGridDims.data(),
      m_generatedGridDomains.data(),
      m_generatedGridScalars.data(),
      m_generatedGridScalars.size());
}

anari::box3 UnstructuredField::bounds() const
{
  return m_bounds;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::SpatialField *);
