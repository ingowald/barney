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

  void markCommitted() override;

  virtual BNScalarField createBarneyScalarField(BNContext context// , int slot
                                                ) const = 0;

  void cleanup()
  {
    if (m_bnField) {
      bnRelease(m_bnField);
      m_bnField = nullptr;
    }
  }
  
  BNScalarField getBarneyScalarField(BNContext context// , int slot
                                     )
  {
    if (!isValid())
      return {};
    // if (!isModelTracked(model, slot)) {
    //   cleanup();
    //   trackModel(model, slot);
    // }
    if (!m_bnField) 
      m_bnField = createBarneyScalarField(context/*,slot*/);
    return m_bnField;
  }
  
  virtual box3 bounds() const = 0;

  BNScalarField m_bnField = 0;
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct UnstructuredField : public SpatialField
{
  UnstructuredField(BarneyGlobalState *s);
  void commit() override;

  BNScalarField createBarneyScalarField(BNContext context// , int slot
                                        ) const;

  box3 bounds() const override;

 private:
  struct Parameters
  {
    helium::IntrusivePtr<helium::Array1D> vertexPosition;
    helium::IntrusivePtr<helium::Array1D> vertexData;
    helium::IntrusivePtr<helium::Array1D> index;
    helium::IntrusivePtr<helium::Array1D> cellType;
    helium::IntrusivePtr<helium::Array1D> cellBegin;
    // "stitcher" extensions
    // helium::IntrusivePtr<helium::ObjectArray> gridData;
    // helium::IntrusivePtr<helium::Array1D> gridDomains;
  } m_params;

  std::vector<math::float4> m_vertices;
  /* barney requires index offsets to be in ascending order, so this
     is an array of the element indices in sequential order of
     elements (because the anari array odesn't have that requirement,
     so let's make sure to fix it) */
  std::vector<int>          m_indices;
  std::vector<int>          m_elementOffsets;
  // std::vector<float> m_generatedVertices;
  // std::vector<int> m_generatedTets;
  // std::vector<int> m_generatedPyrs;
  // std::vector<int> m_generatedWedges;
  // std::vector<int> m_generatedHexes;
  // // for stitcher
  // std::vector<int> m_generatedGridOffsets;
  // std::vector<int> m_generatedGridDims;
  // std::vector<float> m_generatedGridDomains;
  // std::vector<float> m_generatedGridScalars;

  box3 m_bounds;
};

struct BlockStructuredField : public SpatialField
{
  BlockStructuredField(BarneyGlobalState *s);
  void commit() override;

  BNScalarField createBarneyScalarField(BNContext context// , int slot
                                        ) const;

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
  void commit() override;

  BNScalarField createBarneyScalarField(BNContext context// , int slot
                                        ) const;

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
