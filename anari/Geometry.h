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

  void commit() override;
  void markCommitted() override;

  virtual const char *bnSubtype() const = 0;
  virtual void setBarneyParameters(BNGeom geom, BNContext context, int slot) = 0;
  virtual box3 bounds() const = 0;

 protected:
  std::array<helium::IntrusivePtr<Array1D>, 5> m_attributes;
  std::array<helium::IntrusivePtr<Array1D>, 5> m_vertexAttributes;
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct Sphere : public Geometry
{
  Sphere(BarneyGlobalState *s);
  void commit() override;
  bool isValid() const override;

  void setBarneyParameters(BNGeom geom, BNContext context, int slot) override;
  const char *bnSubtype() const override;
  box3 bounds() const override;

 private:
  helium::ChangeObserverPtr<Array1D> m_index;
  helium::ChangeObserverPtr<Array1D> m_vertexPosition;
  helium::ChangeObserverPtr<Array1D> m_vertexRadius;
  float m_globalRadius{0.f};
};

struct Curve : public Geometry
{
  Curve(BarneyGlobalState *s);
  void commit() override;
  bool isValid() const override;
  
  void setBarneyParameters(BNGeom geom, BNContext context, int slot) override;
  const char *bnSubtype() const override;
  box3 bounds() const override;
  
private:
  helium::ChangeObserverPtr<Array1D> m_index;
  helium::ChangeObserverPtr<Array1D> m_vertexPosition;
  helium::ChangeObserverPtr<Array1D> m_vertexRadius;
  float m_globalRadius{0.f};
};

struct Triangle : public Geometry
{
  Triangle(BarneyGlobalState *s);
  void commit() override;
  bool isValid() const override;

  void setBarneyParameters(BNGeom geom, BNContext context, int slot) override;
  const char *bnSubtype() const override;
  box3 bounds() const override;

 private:
  helium::ChangeObserverPtr<Array1D> m_index;
  helium::ChangeObserverPtr<Array1D> m_vertexPosition;
  helium::ChangeObserverPtr<Array1D> m_vertexNormal;
  std::array<helium::IntrusivePtr<Array1D>, 5> m_vertexAttributes;
  std::vector<int> m_generatedIndices;
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Geometry *, ANARI_GEOMETRY);
