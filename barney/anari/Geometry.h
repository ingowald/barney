// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/anari/Array.h"
#include "barney/anari/Object.h"
#include "barney/anari/SpatialField.h"

namespace BARNEY_NS {
  namespace native {
    struct Geometry;
  }
  namespace anari {
    
    struct Geometry : public Object
    {
      Geometry(BarneyGlobalState *s);
      ~Geometry() override;

      /*! implements the anariNewGeometry, and returns a anariGeometry */
      static Geometry *createInstance(std::string_view subtype,
                                      BarneyGlobalState *s);

      void commitParameters() override;
      void markFinalized() override;

      /*! \brief return an instance of the corresponding native geometrby

        \detailed arney native geometries are actually a baked fusion
        of anari::geometry and anari::surface (ie, they are geometries
        but do contain the material and attributes), so each anari
        geometry may possibly get instantiated multiple times from
        different surfaces. To allow this we have the anari::geometry
        be a creator for a (shared pointer to) its corresponding
        native geometry type - each surface can thus create its own
        native instance, and set its material data thereon.  */
      virtual std::shared_ptr<native::Geometry> createNative(BarneyGlobalState *s) = 0;

      /*! set (our) parameters, attributes, etc, on a given instance
          of our corresponding native geometry. this is guaranteed to
          be a native geometry that we have created via createNative,
          but here might be multiple such ones around (one per
          anari::surface that uses this geometry. note this still uses
          the now-deprated barney-native-api's 'BNGeom', but this is
          just a 'native::Geometry*' */
      virtual void setBarneyParameters(BNGeom geom) = 0;
      virtual box3 bounds() const = 0;

    protected:
      virtual void setAttributes(BNGeom geom);
      std::array<math::float4, 5>                  m_constantAttributes;
      std::array<helium::IntrusivePtr<Array1D>, 5> m_primitiveAttributes;
      std::array<helium::IntrusivePtr<Array1D>, 5> m_vertexAttributes;
    };

    // Subtypes /////////////////////////////////////////////////////////////////

    struct Sphere : public Geometry
    {
      Sphere(BarneyGlobalState *s);
      void commitParameters() override;
      void finalize() override;
      bool isValid() const override;

      void setBarneyParameters(BNGeom geom) override;
      std::shared_ptr<native::Geometry> createNative(BarneyGlobalState *s) override;
      box3 bounds() const override;

    private:
      helium::ChangeObserverPtr<Array1D> m_index;
      helium::ChangeObserverPtr<Array1D> m_vertexPosition;
      helium::ChangeObserverPtr<Array1D> m_vertexRadius;
      float m_globalRadius{0.f};
    };

    struct IsoSurface : public Geometry
    {
      IsoSurface(BarneyGlobalState *s);
      void commitParameters() override;
      void finalize() override;
      bool isValid() const override;

      void setBarneyParameters(BNGeom geom) override;
      std::shared_ptr<native::Geometry> createNative(BarneyGlobalState *s) override;
      box3 bounds() const override;

    private:
      helium::ChangeObserverPtr<Array1D> m_isoValues;
      helium::ChangeObserverPtr<SpatialField> m_field;
      float m_isoValue{NAN};
    };

    struct Cylinder : public Geometry
    {
      Cylinder(BarneyGlobalState *s);
      void commitParameters() override;
      void finalize() override;
      bool isValid() const override;

      void setBarneyParameters(BNGeom geom) override;
      std::shared_ptr<native::Geometry> createNative(BarneyGlobalState *s) override;
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
      std::shared_ptr<native::Geometry> createNative(BarneyGlobalState *s) override;
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
      std::shared_ptr<native::Geometry> createNative(BarneyGlobalState *s) override;
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
      std::shared_ptr<native::Geometry> createNative(BarneyGlobalState *s) override;
      box3 bounds() const override;

      void setAttributes(BNGeom geom) override;

    private:
      helium::ChangeObserverPtr<Array1D> m_index;
      helium::ChangeObserverPtr<Array1D> m_vertexPosition;
      helium::ChangeObserverPtr<Array1D> m_vertexNormal;
      helium::IntrusivePtr<Array1D> m_vertexTangent;
      helium::IntrusivePtr<Array1D> m_faceVaryingTangent;
      std::vector<int> m_generatedIndices;
    };

    struct Triangle : public Geometry
    {
      Triangle(BarneyGlobalState *s);
      void commitParameters() override;
      void finalize() override;
      bool isValid() const override;

      void setBarneyParameters(BNGeom geom) override;
      std::shared_ptr<native::Geometry> createNative(BarneyGlobalState *s) override;
      box3 bounds() const override;

      void setAttributes(BNGeom geom) override;
    
    private:
      helium::ChangeObserverPtr<Array1D> m_index;
      helium::ChangeObserverPtr<Array1D> m_vertexPosition;
      helium::ChangeObserverPtr<Array1D> m_vertexNormal;
      helium::IntrusivePtr<Array1D> m_vertexTangent;
      helium::IntrusivePtr<Array1D> m_faceVaryingTangent;
      std::array<helium::IntrusivePtr<Array1D>, 6> m_faceVaryingAttributes;
      std::vector<int> m_generatedIndices;
    };

  }
}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::Geometry *, ANARI_GEOMETRY);
