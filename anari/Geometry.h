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

    static Geometry *createInstance(std::string_view subtype,
                                    BarneyGlobalState *s);

    void commitParameters() override;
    void markFinalized() override;

    virtual const char *bnSubtype() const = 0;
    virtual void setBarneyParameters(BNGeom geom) = 0;
    virtual box3 bounds() const = 0;

  protected:
    void setAttributes(BNGeom geom);
    std::array<math::float4, 5>                  m_constantAttributes;
    std::array<helium::IntrusivePtr<Array1D>, 5> m_primitiveAttributes;
    std::array<helium::IntrusivePtr<Array1D>, 5> m_vertexAttributes;
  };

  // Subtypes ///////////////////////////////////////////////////////////////////

  struct Sphere : public Geometry
  {
    Sphere(BarneyGlobalState *s);
    void commitParameters() override;
    void finalize() override;
    bool isValid() const override;

    void setBarneyParameters(BNGeom geom) override;
    const char *bnSubtype() const override;
    box3 bounds() const override;

  private:
    helium::ChangeObserverPtr<Array1D> m_index;
    helium::ChangeObserverPtr<Array1D> m_vertexPosition;
    helium::ChangeObserverPtr<Array1D> m_vertexRadius;
    float m_globalRadius{0.f};
  };

  struct Cylinder : public Geometry
  {
    Cylinder(BarneyGlobalState *s);
    void commitParameters() override;
    void finalize() override;
    bool isValid() const override;

    void setBarneyParameters(BNGeom geom) override;
    const char *bnSubtype() const override;
    box3 bounds() const override;

  private:
    helium::ChangeObserverPtr<Array1D> m_index;
    helium::ChangeObserverPtr<Array1D> m_radius;
    helium::ChangeObserverPtr<Array1D> m_vertexPosition;
    float m_globalRadius{0.f};
    std::vector<math::uint2> m_generatedIndices;
    std::vector<float> m_generatedRadii;
  };

  struct Cone : public Geometry
  {
    Cone(BarneyGlobalState *s);
    void commitParameters() override;
    void finalize() override;
    bool isValid() const override;

    void setBarneyParameters(BNGeom geom) override;
    const char *bnSubtype() const override;
    box3 bounds() const override;

  private:
    helium::ChangeObserverPtr<Array1D> m_index;
    helium::ChangeObserverPtr<Array1D> m_vertexPosition;
    helium::ChangeObserverPtr<Array1D> m_vertexRadius;
    std::vector<math::uint2> m_generatedIndices;
  };

  struct Curve : public Geometry
  {
    Curve(BarneyGlobalState *s);
    void commitParameters() override;
    void finalize() override;
    bool isValid() const override;

    void setBarneyParameters(BNGeom geom) override;
    const char *bnSubtype() const override;
    box3 bounds() const override;

  private:
    helium::ChangeObserverPtr<Array1D> m_index;
    helium::ChangeObserverPtr<Array1D> m_vertexPosition;
    helium::ChangeObserverPtr<Array1D> m_vertexRadius;
    float m_globalRadius{0.f};
  };

  struct Quad : public Geometry
  {
    Quad(BarneyGlobalState *s);
    void commitParameters() override;
    void finalize() override;
    bool isValid() const override;

    void setBarneyParameters(BNGeom geom) override;
    const char *bnSubtype() const override;
    box3 bounds() const override;

  private:
    helium::ChangeObserverPtr<Array1D> m_index;
    helium::ChangeObserverPtr<Array1D> m_vertexPosition;
    helium::ChangeObserverPtr<Array1D> m_vertexNormal;
    std::vector<int> m_generatedIndices;
  };

  struct Triangle : public Geometry
  {
    Triangle(BarneyGlobalState *s);
    void commitParameters() override;
    void finalize() override;
    bool isValid() const override;

    void setBarneyParameters(BNGeom geom) override;
    const char *bnSubtype() const override;
    box3 bounds() const override;

  private:
    helium::ChangeObserverPtr<Array1D> m_index;
    helium::ChangeObserverPtr<Array1D> m_vertexPosition;
    helium::ChangeObserverPtr<Array1D> m_vertexNormal;
    std::vector<int> m_generatedIndices;
  };

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Geometry *, ANARI_GEOMETRY);
