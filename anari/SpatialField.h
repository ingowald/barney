// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari
#include "helium/array/Array1D.h"
#include "helium/array/ObjectArray.h"
// ours
#include "Object.h"

namespace barney_device {

struct SpatialField : public Object
{
  SpatialField(BarneyGlobalState *s);
  ~SpatialField() override;

  static SpatialField *createInstance(
      std::string_view subtype, BarneyGlobalState *s);

  void markCommitted() override;

  virtual BNScalarField makeBarneyScalarField(BNDataGroup dg) const = 0;

  virtual anari::box3 bounds() const = 0;
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct UnstructuredField : public SpatialField
{
  UnstructuredField(BarneyGlobalState *s);
  void commit() override;

  BNScalarField makeBarneyScalarField(BNDataGroup dg) const;

  anari::box3 bounds() const override;

  struct Parameters
  {
    helium::IntrusivePtr<helium::Array1D> vertexPosition;
    helium::IntrusivePtr<helium::Array1D> vertexData;
    helium::IntrusivePtr<helium::Array1D> index;
    helium::IntrusivePtr<helium::Array1D> cellIndex;
    // "stitcher" extensions
    helium::IntrusivePtr<helium::ObjectArray> gridData;
    helium::IntrusivePtr<helium::Array1D> gridDomains;
  } m_params;

  std::vector<float> m_generatedVertices;
  std::vector<int> m_generatedTets;
  std::vector<int> m_generatedPyrs;
  std::vector<int> m_generatedWedges;
  std::vector<int> m_generatedHexes;
  // for stitcher
  std::vector<int> m_generatedGridOffsets;
  std::vector<int> m_generatedGridDims;
  std::vector<float> m_generatedGridDomains;
  std::vector<float> m_generatedGridScalars;

  anari::box3 m_bounds;
};

struct BlockStructuredField : public SpatialField
{
  BlockStructuredField(BarneyGlobalState *s);
  void commit() override;

  BNScalarField makeBarneyScalarField(BNDataGroup dg) const;

  anari::box3 bounds() const override;

  struct Parameters
  {
    helium::IntrusivePtr<helium::Array1D> cellWidth;
    helium::IntrusivePtr<helium::Array1D> blockBounds;
    helium::IntrusivePtr<helium::Array1D> blockLevel;
    helium::IntrusivePtr<helium::ObjectArray> blockData;
  } m_params;

  std::vector<float> m_generatedCellWidths;
  std::vector<int> m_generatedBlockBounds;
  std::vector<int> m_generatedBlockLevels;
  std::vector<int> m_generatedBlockOffsets;
  std::vector<float> m_generatedBlockScalars;

  anari::box3 m_bounds;
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(
    barney_device::SpatialField *, ANARI_SPATIAL_FIELD);
