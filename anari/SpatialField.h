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

    static SpatialField *createInstance(std::string_view subtype,
                                        BarneyGlobalState *s);

    void markFinalized() override;

    virtual BNScalarField createBarneyScalarField() const = 0;

    void cleanup()
    {
      if (m_bnField) {
        bnRelease(m_bnField);
        m_bnField = nullptr;
      }
    }

    BNScalarField getBarneyScalarField()
    {
      if (!isValid())
        return {};
      if (!m_bnField)
        m_bnField = createBarneyScalarField();
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

    BNScalarField createBarneyScalarField() const override;

    box3 bounds() const override;
    bool isValid() const override;

  private:
    struct Parameters
    {
      Parameters(helium::BaseObject *observer)
        : vertexPosition(observer),
          vertexData(observer),
          cellData(observer),
          index(observer),
          cellType(observer),
          cellBegin(observer)
      {}
      helium::ChangeObserverPtr<helium::Array1D> vertexPosition;
      helium::ChangeObserverPtr<helium::Array1D> vertexData;
      helium::ChangeObserverPtr<helium::Array1D> cellData;
      helium::ChangeObserverPtr<helium::Array1D> index;
      helium::ChangeObserverPtr<helium::Array1D> cellType;
      helium::ChangeObserverPtr<helium::Array1D> cellBegin;
    } m_params;

    struct BarneyData
    {
      BNData vertices{nullptr};
      BNData scalars{nullptr};
      BNData indices{nullptr};
      BNData cellType{nullptr};
      BNData elementOffsets{nullptr};
    } m_bnData;

    box3 m_bounds;
  };

  struct BlockStructuredField : public SpatialField
  {
    BlockStructuredField(BarneyGlobalState *s);
    void commitParameters() override;
    void finalize() override;

    BNScalarField createBarneyScalarField() const override;

    box3 bounds() const override;

    struct Parameters
    {
      Parameters(helium::BaseObject *observer)
        : refinementRatio(observer),
          blockBounds(observer),
          blockLevel(observer),
          data(observer)
      {}
      helium::ChangeObserverPtr<helium::Array1D> refinementRatio;
      helium::ChangeObserverPtr<helium::Array1D> blockBounds;
      helium::ChangeObserverPtr<helium::Array1D> blockLevel;
      helium::ChangeObserverPtr<helium::Array1D> data;
    } m_params;

    struct BarneyData
    {
      BNData scalars{nullptr};
      BNData blockOrigins{nullptr};
      BNData blockDims{nullptr};
      BNData blockLevels{nullptr};
      BNData blockOffsets{nullptr};
      BNData levelRefinements{nullptr};
    } m_bnData;

    std::vector<math::int3> m_generatedBlockOrigins;
    std::vector<math::int3> m_generatedBlockDims;
    std::vector<int> m_generatedBlockLevels;
    std::vector<uint64_t> m_generatedBlockOffsets;
    std::vector<int> m_generatedRefinements;

    box3 m_bounds;
  };

  struct StructuredRegularField : public SpatialField
  {
    StructuredRegularField(BarneyGlobalState *s);
    void commitParameters() override;
    void finalize() override;

    BNScalarField createBarneyScalarField() const override;

    box3 bounds() const override;
    bool isValid() const override;

    math::uint3 m_dims{0u};
    math::float3 m_origin;
    math::float3 m_spacing;
    math::float3 m_coordUpperBound;

    helium::ChangeObserverPtr<helium::Array3D> m_data;
  };

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::SpatialField *,
                                    ANARI_SPATIAL_FIELD);
