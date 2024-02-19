// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Array.h"
#include "Object.h"

namespace barney_device {

struct Geometry : public Object
{
  Geometry(BarneyGlobalState *s);
  ~Geometry() override;

  static Geometry *createInstance(
      std::string_view subtype, BarneyGlobalState *s);

  void markCommitted() override;

  virtual BNGeom makeBarneyGeometry(
      BNDataGroup dg, const BNMaterialHelper *material) const = 0;

  virtual box3 bounds() const = 0;
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct Sphere : public Geometry
{
  Sphere(BarneyGlobalState *s);
  void commit() override;
    
  BNGeom makeBarneyGeometry(
      BNDataGroup dg, const BNMaterialHelper *material) const override;

  box3 bounds() const override;

  bool isValid() const override;

 private:
  void cleanup();

  helium::IntrusivePtr<Array1D> m_index;
  helium::IntrusivePtr<Array1D> m_vertexPosition;
  helium::IntrusivePtr<Array1D> m_vertexRadius;
  float m_globalRadius{0.f};

  std::vector<int> m_generatedIndices;
};

struct Triangle : public Geometry
{
  Triangle(BarneyGlobalState *s);
  void commit() override;

  BNGeom makeBarneyGeometry(
      BNDataGroup dg, const BNMaterialHelper *material) const override;

  box3 bounds() const override;

  bool isValid() const override;

 private:
  void cleanup();

  helium::IntrusivePtr<Array1D> m_index;
  helium::IntrusivePtr<Array1D> m_vertexPosition;

  std::vector<int> m_generatedIndices;
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Geometry *, ANARI_GEOMETRY);
