// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

// std
#include <cfloat>
// ours
#include "Array.h"
#include "SpatialField.h"

namespace barney_device {

SpatialField::SpatialField(BarneyGlobalState *s)
    : Object(ANARI_SPATIAL_FIELD, s)
{}

SpatialField::~SpatialField() = default;

SpatialField *SpatialField::createInstance(
    std::string_view subtype, BarneyGlobalState *s)
{
  if (subtype == "unstructured")
    return new UnstructuredField(s);
  else if (subtype == "amr")
    return new BlockStructuredField(s);
  else if (subtype == "structuredRegular")
    return new StructuredRegularField(s);
  else
    return (SpatialField *)new UnknownObject(ANARI_SPATIAL_FIELD, s);
}

void SpatialField::markCommitted()
{
  deviceState()->markSceneChanged();
  Object::markCommitted();
}

// Subtypes ///////////////////////////////////////////////////////////////////

// StructuredRegularField //

StructuredRegularField::StructuredRegularField(BarneyGlobalState *s)
    : SpatialField(s)
{}

void StructuredRegularField::commit()
{
  Object::commit();
  m_data = getParamObject<helium::Array3D>("data");

  if (!m_data) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'data' on 'structuredRegular' field");
    return;
  }

  m_origin = getParam<helium::float3>("origin", helium::float3(0.f));
  m_spacing = getParam<helium::float3>("spacing", helium::float3(1.f));
  m_dims = m_data->size();

  const auto dims = m_data->size();
  m_coordUpperBound = helium::float3(std::nextafter(dims.x - 1, 0),
      std::nextafter(dims.y - 1, 0),
      std::nextafter(dims.z - 1, 0));
}

bool StructuredRegularField::isValid() const
{
  return m_data;
}

  BNScalarField StructuredRegularField::createBarneyScalarField(
    BNContext context// , int slot
                                                              ) const
{
  if (!isValid())
    return {};
  auto ctx = deviceState()->context;
  BNDataType barneyType;
  switch (m_data->elementType()) {
  case ANARI_FLOAT32:
    barneyType = BN_FLOAT;
    break;
  case ANARI_UINT8:
    barneyType = BN_UFIXED8;
    break;
  // case ANARI_FLOAT64:
  //   return ((double *)m_data)[i];
  // case ANARI_UFIXED8:
  //   return ((uint8_t *)m_data)[i] /
  //   float(std::numeric_limits<uint8_t>::max());
  // case ANARI_UFIXED16:
  //   return ((uint16_t *)m_data)[i]
  //       / float(std::numeric_limits<uint16_t>::max());
  // case ANARI_FIXED16:
  //   return ((int16_t *)m_data)[i] /
  //   float(std::numeric_limits<int16_t>::max());
  default:
    throw std::runtime_error("scalar type not implemented ...");
  }
  auto dims = m_data->size();
  auto field = bnStructuredDataCreate(context,
                                      0/*slot*/,
      (const int3 &)dims,
      barneyType,
      m_data->data(),
      (const float3 &)m_origin,
      (const float3 &)m_spacing);
  return field;
}

box3 StructuredRegularField::bounds() const
{
  return isValid()
      ? box3(m_origin, m_origin + ((helium::float3(m_dims) - 1.f) * m_spacing))
      : box3{};
}

// UnstructuredField //

UnstructuredField::UnstructuredField(BarneyGlobalState *s) : SpatialField(s) {}

void UnstructuredField::commit()
{
  Object::commit();

  m_params.vertexPosition = getParamObject<helium::Array1D>("vertex.position");
  m_params.vertexData = getParamObject<helium::Array1D>("vertex.data");
  m_params.index = getParamObject<helium::Array1D>("index");
  m_params.cellType = getParamObject<helium::Array1D>("cell.type");
  m_params.cellBegin = getParamObject<helium::Array1D>("cell.begin");

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

  if (!m_params.cellType) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'cell.type' on unstructured spatial field");
    return;
  }

  if (!m_params.cellBegin) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'cell.begin' on unstructured spatial field");
    return;
  }

  // m_params.gridData = getParamObject<helium::ObjectArray>("grid.data");
  // m_params.gridDomains = getParamObject<helium::Array1D>("grid.domains");

  m_bounds.invalidate();
  m_indices.clear();
  m_vertices.clear();
  m_elementOffsets.clear();
  
  auto *vertexPosition = m_params.vertexPosition->beginAs<math::float3>();
  int numVertices      = m_params.vertexPosition->endAs<math::float3>()-vertexPosition;
  auto *vertexData = m_params.vertexData->beginAs<float>();
  m_vertices.resize(numVertices);
  for (int i=0;i<numVertices;i++) {
    (math::float3&)m_vertices[i] = vertexPosition[i];
    m_bounds.insert(vertexPosition[i]);
    m_vertices[i].w = vertexData[i];
  }
  auto *index = m_params.index->beginAs<uint32_t>();
  auto *cellBegin = m_params.cellBegin->beginAs<uint32_t>();
  auto *cellType = m_params.cellType->beginAs<uint8_t>();

  // size_t numVerts = m_params.vertexPosition->size();
  size_t numCells = m_params.cellType->size(); //endAs<uint64_t>() - index;
  // this isn't fully spec'ed yet
  enum { _ANARI_TET = 0, _ANARI_HEX=1, _ANARI_WEDGE=2, _ANARI_PYR=3 };
  enum { _VTK_TET = 10, _VTK_HEX=12, _VTK_WEDGE=13, _VTK_PYR=14 };
  for (int cellIdx=0;cellIdx<(int)numCells;cellIdx++) {
    int thisOffset = m_indices.size();
    m_elementOffsets.push_back(thisOffset);
    uint8_t type = cellType[cellIdx];
    int numToCopy = -1;
    switch(type) {
    case _ANARI_TET:
      numToCopy = 4; break;
    case _ANARI_HEX:
      numToCopy = 8; break;
    case _ANARI_WEDGE:
      numToCopy = 6; break;
    case _ANARI_PYR:
      numToCopy = 5; break;
    case _VTK_TET:
      numToCopy = 4; break;
    case _VTK_HEX:
      numToCopy = 8; break;
    case _VTK_WEDGE:
      numToCopy = 6; break;
    case _VTK_PYR:
      numToCopy = 5; break;
    default:
      throw std::runtime_error("buggy/invalid unstructured element type!?");
    };
    int inputBegin = cellBegin[cellIdx];
    for (int i=0;i<numToCopy;i++) {
      m_indices.push_back(index[inputBegin+i]);
    }
  }
}

BNScalarField UnstructuredField::createBarneyScalarField(
    BNContext context// , int slot
                                                         ) const
{
  // auto ctx = deviceState()->context;
  std::cout << "==================================================================" << std::endl;
  std::cout << "BANARI: CREATING UMESH OF " << m_elementOffsets.size() << " elements" << std::endl;
  std::cout << "==================================================================" << std::endl;
  return bnUMeshCreate(context,
                       0/*slot*/,
                       (const ::float4 *)m_vertices.data(),
                       m_vertices.size(),
                       m_indices.data(),
                       m_indices.size(),
                       m_elementOffsets.data(),
                       m_elementOffsets.size(),
                       // m_generatedVertices.data(),
                       // m_generatedVertices.size() / 4,
                       // m_generatedTets.data(),
                       // m_generatedTets.size() / 4,
                       // m_generatedPyrs.data(),
                       // m_generatedPyrs.size() / 5,
                       // m_generatedWedges.data(),
                       // m_generatedWedges.size() / 6,
                       // m_generatedHexes.data(),
                       // m_generatedHexes.size() / 8,
                       // m_generatedGridOffsets.size(),
                       // m_generatedGridOffsets.data(),
                       // m_generatedGridDims.data(),
                       // m_generatedGridDomains.data(),
                       // m_generatedGridScalars.data(),
                       // m_generatedGridScalars.size()
                       nullptr
                       );
}

box3 UnstructuredField::bounds() const
{
  return m_bounds;
}

// BlockStructuredField //

BlockStructuredField::BlockStructuredField(BarneyGlobalState *s)
    : SpatialField(s)
{}

void BlockStructuredField::commit()
{
  Object::commit();

  m_params.cellWidth = getParamObject<helium::Array1D>("cellWidth");
  m_params.blockBounds = getParamObject<helium::Array1D>("block.bounds");
  m_params.blockLevel = getParamObject<helium::Array1D>("block.level");
  m_params.blockData = getParamObject<helium::ObjectArray>("block.data");

  if (!m_params.blockBounds) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'block.bounds' on amr spatial field");
    return;
  }

  if (!m_params.blockLevel) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'block.level' on amr spatial field");
    return;
  }

  if (!m_params.blockData) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'block.data' on amr spatial field");
    return;
  }

  size_t numBlocks = m_params.blockData->totalSize();
  auto *blockBounds = m_params.blockBounds->beginAs<box3i>();
  auto *blockLevels = m_params.blockLevel->beginAs<int>();
  auto *blockData = (helium::Array3D **)m_params.blockData->handlesBegin();

  m_generatedBlockBounds.clear();
  m_generatedBlockLevels.clear();
  m_generatedBlockOffsets.clear();
  m_generatedBlockScalars.clear();

  m_bounds.invalidate();

  for (size_t i = 0; i < numBlocks; ++i) {
    const box3i bounds = *(blockBounds + i);
    const int level = *(blockLevels + i);
    const helium::Array3D *bd = *(blockData + i);

    m_generatedBlockBounds.push_back(bounds.lower.x);
    m_generatedBlockBounds.push_back(bounds.lower.y);
    m_generatedBlockBounds.push_back(bounds.lower.z);
    m_generatedBlockBounds.push_back(bounds.upper.x);
    m_generatedBlockBounds.push_back(bounds.upper.y);
    m_generatedBlockBounds.push_back(bounds.upper.z);
    m_generatedBlockLevels.push_back(level);
    m_generatedBlockOffsets.push_back(m_generatedBlockScalars.size());

    for (unsigned z = 0; z < bd->size().z; ++z)
      for (unsigned y = 0; y < bd->size().y; ++y)
        for (unsigned x = 0; x < bd->size().x; ++x) {
          size_t index =
              z * size_t(bd->size().x) * bd->size().y + y * bd->size().x + x;
          float f = bd->dataAs<float>()[index];
          m_generatedBlockScalars.push_back(f);
        }

    box3 worldBounds;
    worldBounds.lower = math::float3(bounds.lower.x * (1 << level),
        bounds.lower.y * (1 << level),
        bounds.lower.z * (1 << level));
    worldBounds.upper = math::float3((bounds.upper.x + 1) * (1 << level),
        (bounds.upper.y + 1) * (1 << level),
        (bounds.upper.z + 1) * (1 << level));
    m_bounds.insert(worldBounds);
  }
}

BNScalarField BlockStructuredField::createBarneyScalarField(
    BNContext context// , int slot
                                                            ) const
{
  return bnBlockStructuredAMRCreate(context,
                                    0/*slot*/,
      m_generatedBlockBounds.data(),
      m_generatedBlockBounds.size() / 6,
      m_generatedBlockLevels.data(),
      m_generatedBlockOffsets.data(),
      m_generatedBlockScalars.data(),
      m_generatedBlockScalars.size());
}

box3 BlockStructuredField::bounds() const
{
  return m_bounds;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::SpatialField *);
