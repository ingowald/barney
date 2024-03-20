// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "common.h"
#include "Geometry.h"
// std
#include <cassert>
#include <numeric>

namespace barney_device {

Geometry::Geometry(BarneyGlobalState *s) : Object(ANARI_GEOMETRY, s) {}

Geometry::~Geometry() = default;

void Geometry::commit()
{
  m_attributes[0] = getParamObject<Array1D>("primitive.attribute0");
  m_attributes[1] = getParamObject<Array1D>("primitive.attribute1");
  m_attributes[2] = getParamObject<Array1D>("primitive.attribute2");
  m_attributes[3] = getParamObject<Array1D>("primitive.attribute3");
  m_attributes[4] = getParamObject<Array1D>("primitive.color");

  for (int i = 0; i < 5; ++i) {
    if (m_attributes[i]) {
      size_t sizeInBytes
          = m_attributes[i]->size() * anari::sizeOf(m_attributes[i]->elementType());
    }
  }
}

Geometry *Geometry::createInstance(
    std::string_view subtype, BarneyGlobalState *s)
{
  if (subtype == "sphere")
    return new Sphere(s);
  else if (subtype == "triangle")
    return new Triangle(s);
  else
    return (Geometry *)new UnknownObject(ANARI_GEOMETRY, s);
}

void Geometry::markCommitted()
{
  deviceState()->markSceneChanged();
  Object::markCommitted();
}

// Subtypes ///////////////////////////////////////////////////////////////////

Sphere::Sphere(BarneyGlobalState *s) : Geometry(s) {}

void Sphere::commit()
{
  Geometry::commit();

  cleanup();

  m_index = getParamObject<Array1D>("primitive.index");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  m_vertexRadius = getParamObject<Array1D>("vertex.radius");

  m_globalRadius = getParam<float>("radius", 0.01f);

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on sphere geometry");
    return;
  }

  if (m_index) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "primitive.index parameter on sphere geometry not yet supported");
  }

  m_generatedIndices.clear();

  m_vertexPosition->addCommitObserver(this);
  if (m_vertexRadius)
    m_vertexRadius->addCommitObserver(this);
}

BNGeom Sphere::makeBarneyGeometry(
    BNModel model, int slot, const BNMaterial material) const
{
  BNGeom geom = bnGeometryCreate(model, slot, "spheres");
  BNData origins = bnDataCreate(model,
      slot,
      BN_FLOAT3,
      m_vertexPosition->totalSize(),
      (const float3 *)m_vertexPosition->data());
  bnSetData(geom, "origins", origins);
  if (m_vertexRadius) {
    BNData radii = bnDataCreate(model,
        slot,
        BN_FLOAT,
        m_vertexRadius->totalSize(),
        m_vertexRadius->dataAs<float>());
    bnSetData(geom, "radii", radii);
  } else
    bnSet1f(geom, "radius", m_globalRadius);
  bnSetObject(geom, "material", material);
  bnCommit(geom);
  return geom;
}

box3 Sphere::bounds() const
{
  if (!isValid())
    return {};

  box3 result;
  if (m_index) {
    std::for_each(m_index->beginAs<uint32_t>(),
        m_index->beginAs<uint32_t>() + m_index->totalSize(),
        [&](uint32_t index) {
          math::float3 v = *(m_vertexPosition->beginAs<math::float3>() + index);
          float r = m_vertexRadius ?
              *(m_vertexRadius->beginAs<float>() + index) : m_globalRadius;
          result.insert(math::float3{v.x - r, v.y - r, v.z - r});
          result.insert(math::float3{v.x + r, v.y + r, v.z + r});
        });
  } else {
    for (size_t i = 0; i < m_vertexPosition->totalSize(); ++i) {
      math::float3 v = *(m_vertexPosition->beginAs<math::float3>() + i);
      float r = m_vertexRadius ?
          *(m_vertexRadius->beginAs<float>() + i) : m_globalRadius;
      result.insert(math::float3{v.x - r, v.y - r, v.z - r});
      result.insert(math::float3{v.x + r, v.y + r, v.z + r});
    }
  }
  return result;
}

size_t Sphere::numRequiredGPUBytes() const
{
  return getNumBytes(m_vertexPosition) + getNumBytes(m_vertexRadius);
}

bool Sphere::isValid() const
{
  return m_vertexPosition;
}

void Sphere::cleanup()
{
  if (m_index)
    m_index->removeCommitObserver(this);
  if (m_vertexPosition)
    m_vertexPosition->removeCommitObserver(this);
  if (m_vertexRadius)
    m_vertexRadius->removeCommitObserver(this);
}

Triangle::Triangle(BarneyGlobalState *s) : Geometry(s) {}

void Triangle::commit()
{
  Geometry::commit();

  cleanup();

  m_index = getParamObject<Array1D>("primitive.index");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on triangle geometry");
    return;
  }

  m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
  m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
  m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
  m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
  m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");

  m_generatedIndices.clear();

  m_vertexPosition->addCommitObserver(this);
  if (m_index) {
    m_index->addCommitObserver(this);
  } else {
    m_generatedIndices.resize(m_vertexPosition->totalSize());
    std::iota(m_generatedIndices.begin(), m_generatedIndices.end(), 0);
  }
}

static void addAttribute(BNGeom geom, BNModel model, int slot,
                         const helium::IntrusivePtr<Array1D> &attribute,
                         std::string name)
{
  if (attribute) {
    BNData attr = makeBarneyData(model, slot, attribute);
    
    if (attr) {
      bnSetData(geom, name.c_str(), attr);
    }
  }
}

BNGeom Triangle::makeBarneyGeometry(
    BNModel model, int slot, const BNMaterial material) const
{
  BNGeom geom = bnGeometryCreate(model, slot, "triangles");

  int numVertices = m_vertexPosition->totalSize();
  int numIndices = m_index ? m_index->size() : (m_generatedIndices.size() / 3);
  const float3 *vertices = (const float3 *)m_vertexPosition->data();
  const int3 *indices = m_index ? (const int3 *)m_index->data()
                                : (const int3 *)m_generatedIndices.data();

  BNData _vertices = bnDataCreate(model, slot, BN_FLOAT3, numVertices, vertices);
  bnSetAndRelease(geom, "vertices" ,_vertices);
  
  BNData _indices  = bnDataCreate(model, slot, BN_INT3, numIndices, indices);
  bnSetAndRelease(geom, "indices", _indices);

  bnSetObject(geom, "material", material);

  addAttribute(geom, model, slot, m_attributes[0], "primitive.attribute0");
  addAttribute(geom, model, slot, m_attributes[1], "primitive.attribute1");
  addAttribute(geom, model, slot, m_attributes[2], "primitive.attribute2");
  addAttribute(geom, model, slot, m_attributes[3], "primitive.attribute3");
  addAttribute(geom, model, slot, m_attributes[4], "primitive.attribute4");

  addAttribute(geom, model, slot, m_vertexAttributes[0], "vertex.attribute0");
  addAttribute(geom, model, slot, m_vertexAttributes[1], "vertex.attribute1");
  addAttribute(geom, model, slot, m_vertexAttributes[2], "vertex.attribute2");
  addAttribute(geom, model, slot, m_vertexAttributes[3], "vertex.attribute3");
  addAttribute(geom, model, slot, m_vertexAttributes[4], "vertex.attribute4");

  bnCommit(geom);

  return geom;
}

box3 Triangle::bounds() const
{
  if (!isValid())
    return {};

  box3 result;
  if (m_index) {
    std::for_each(m_index->beginAs<math::uint3>(),
        m_index->beginAs<math::uint3>() + m_index->totalSize(),
        [&](math::uint3 index) {
          math::float3 v1 =
              *(m_vertexPosition->beginAs<math::float3>() + index.x);
          math::float3 v2 =
              *(m_vertexPosition->beginAs<math::float3>() + index.y);
          math::float3 v3 =
              *(m_vertexPosition->beginAs<math::float3>() + index.z);
          result.insert(v1);
          result.insert(v2);
          result.insert(v3);
        });
  } else {
    std::for_each(m_vertexPosition->beginAs<math::float3>(),
        m_vertexPosition->beginAs<math::float3>()
            + m_vertexPosition->totalSize(),
        [&](math::float3 v) { result.insert(v); });
  }
  return result;
}

size_t Triangle::numRequiredGPUBytes() const
{
  return getNumBytes(m_vertexPosition) + getNumBytes(m_index);
}

bool Triangle::isValid() const
{
  return m_vertexPosition;
}

void Triangle::cleanup()
{
  if (m_index)
    m_index->removeCommitObserver(this);
  if (m_vertexPosition)
    m_vertexPosition->removeCommitObserver(this);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Geometry *);
