// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari
#include "helium/array/Array1D.h"
#include "helium/array/Array3D.h"
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

  void markFinalized() override;

  virtual BNScalarField createBarneyScalarField(BNContext context) const = 0;

  void cleanup()
  {
    if (m_bnField) {
      bnRelease(m_bnField);
      m_bnField = nullptr;
    }
  }

  BNScalarField getBarneyScalarField(BNContext context)
  {
    if (!isValid())
      return {};
    if (!m_bnField)
      m_bnField = createBarneyScalarField(context);
    return m_bnField;
  }

  virtual box3 bounds() const = 0;

  BNScalarField m_bnField = 0;
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct UnstructuredField : public SpatialField
{
  UnstructuredField(BarneyGlobalState *s);

  void commitParameters() override;
  void finalize() override;

  BNScalarField createBarneyScalarField(BNContext context) const override;

  box3 bounds() const override;

 private:
  struct Parameters
  {
    helium::IntrusivePtr<helium::Array1D> vertexPosition;
    helium::IntrusivePtr<helium::Array1D> vertexData;
    helium::IntrusivePtr<helium::Array1D> index;
    helium::IntrusivePtr<helium::Array1D> cellType;
    helium::IntrusivePtr<helium::Array1D> cellBegin;
  } m_params;

  std::vector<math::float4> m_vertices;
  /* barney requires index offsets to be in ascending order, so this
     is an array of the element indices in sequential order of
     elements (because the anari array odesn't have that requirement,
     so let's make sure to fix it) */
  std::vector<int> m_indices;
  std::vector<int> m_elementOffsets;

  box3 m_bounds;
};

struct BlockStructuredField : public SpatialField
{
  BlockStructuredField(BarneyGlobalState *s);
  void commitParameters() override;
  void finalize() override;

  BNScalarField createBarneyScalarField(BNContext context) const override;

  box3 bounds() const override;

  struct Parameters
  {
    helium::IntrusivePtr<helium::Array1D> cellWidth;
    helium::IntrusivePtr<helium::Array1D> blockBounds;
    helium::IntrusivePtr<helium::Array1D> blockLevel;
    helium::IntrusivePtr<helium::ObjectArray> blockData;
  } m_params;

  std::vector<int> m_generatedBlockBounds;
  std::vector<int> m_generatedBlockLevels;
  std::vector<int> m_generatedBlockOffsets;
  std::vector<float> m_generatedBlockScalars;

  box3 m_bounds;
};

struct StructuredRegularField : public SpatialField
{
  StructuredRegularField(BarneyGlobalState *s);
  void commitParameters() override;
  void finalize() override;

  BNScalarField createBarneyScalarField(BNContext context) const override;

  box3 bounds() const override;
  bool isValid() const override;

  math::uint3 m_dims{0u};
  math::float3 m_origin;
  math::float3 m_spacing;
  math::float3 m_coordUpperBound;

  std::vector<float> m_generatedCellWidths;
  std::vector<int> m_generatedBlockBounds;
  std::vector<int> m_generatedBlockLevels;
  std::vector<int> m_generatedBlockOffsets;
  std::vector<float> m_generatedBlockScalars;

  helium::IntrusivePtr<helium::Array3D> m_data;
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(
    barney_device::SpatialField *, ANARI_SPATIAL_FIELD);
