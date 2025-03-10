// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Geometry.h"
#include "common.h"
// std
#include <cassert>
#include <iostream>
#include <numeric>

namespace barney_device {

// Helper functions ///////////////////////////////////////////////////////////

static void addAttribute(BNGeom geom,
    BNContext context,
    // int slot,
    const helium::IntrusivePtr<Array1D> &attribute,
    std::string name)
{
  if (!attribute)
    return;

  BNData attr = makeBarneyData(context, 0 /*slot*/, attribute);
  if (attr)
    bnSetData(geom, name.c_str(), attr);
}

// Base Geometry definitions //////////////////////////////////////////////////

Geometry::Geometry(BarneyGlobalState *s) : Object(ANARI_GEOMETRY, s) {}

Geometry::~Geometry() = default;

Geometry *Geometry::createInstance(
    std::string_view subtype, BarneyGlobalState *s)
{
  if (subtype == "sphere")
    return new Sphere(s);
  if (subtype == "curve")
    return new Curve(s);
  if (subtype == "triangle")
    return new Triangle(s);
  if (subtype == "triangles")
    std::cerr
        << "#banari: WARNING - you tried to created 'triangle*s*' geometry, but ANARI terminology is 'triangle'. This is almost certainly an error"
        << std::endl;
  if (subtype == "curves")
    std::cerr
        << "#banari: WARNING - you tried to created 'curve*s*' geometry, but ANARI terminology is 'curve'. This is almost certainly an error"
        << std::endl;
  if (subtype == "spheres")
    std::cerr
        << "#banari: WARNING - you tried to created 'sphere*s*' geometry, but ANARI terminology is 'sphere'. This is almost certainly an error"
        << std::endl;
  return (Geometry *)new UnknownObject(ANARI_GEOMETRY, s);
}

void Geometry::commitParameters()
{
  m_attributes[0] = getParamObject<Array1D>("primitive.attribute0");
  m_attributes[1] = getParamObject<Array1D>("primitive.attribute1");
  m_attributes[2] = getParamObject<Array1D>("primitive.attribute2");
  m_attributes[3] = getParamObject<Array1D>("primitive.attribute3");
  m_attributes[4] = getParamObject<Array1D>("primitive.color");

  m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
  m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
  m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
  m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
  m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");
}

void Geometry::markFinalized()
{
  deviceState()->markSceneChanged();
  Object::markFinalized();
}

// Subtypes ///////////////////////////////////////////////////////////////////

// Sphere //

Sphere::Sphere(BarneyGlobalState *s)
    : Geometry(s), m_index(this), m_vertexPosition(this), m_vertexRadius(this)
{}

void Sphere::commitParameters()
{
  Geometry::commitParameters();
  m_index = getParamObject<Array1D>("primitive.index");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  m_vertexRadius = getParamObject<Array1D>("vertex.radius");
  m_globalRadius = getParam<float>("radius", 0.01f);
}

void Sphere::finalize()
{
  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on sphere geometry");
    return;
  }
}

void Sphere::setBarneyParameters(BNGeom geom, BNContext context)
{
  int slot = 0;
  BNData origins = bnDataCreate(context,
      slot,
      BN_FLOAT3,
      m_vertexPosition->totalSize(),
      (const bn_float3 *)m_vertexPosition->data());
  bnSetData(geom, "origins", origins);
  if (m_vertexRadius) {
    BNData radii = bnDataCreate(context,
        slot,
        BN_FLOAT,
        m_vertexRadius->totalSize(),
        m_vertexRadius->dataAs<float>());
    bnSetData(geom, "radii", radii);
  } else
    bnSet1f(geom, "radius", m_globalRadius);

  addAttribute(geom, context, m_vertexAttributes[0], "vertex.attribute0");
  addAttribute(geom, context, m_vertexAttributes[1], "vertex.attribute1");
  addAttribute(geom, context, m_vertexAttributes[2], "vertex.attribute2");
  addAttribute(geom, context, m_vertexAttributes[3], "vertex.attribute3");
  addAttribute(geom, context, m_vertexAttributes[4], "vertex.color");
}

bool Sphere::isValid() const
{
  return m_vertexPosition;
}

const char *Sphere::bnSubtype() const
{
  return "spheres";
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
          float r = m_vertexRadius ? *(m_vertexRadius->beginAs<float>() + index)
                                   : m_globalRadius;
          result.insert(math::float3{v.x - r, v.y - r, v.z - r});
          result.insert(math::float3{v.x + r, v.y + r, v.z + r});
        });
  } else {
    for (size_t i = 0; i < m_vertexPosition->totalSize(); ++i) {
      math::float3 v = *(m_vertexPosition->beginAs<math::float3>() + i);
      float r = m_vertexRadius ? *(m_vertexRadius->beginAs<float>() + i)
                               : m_globalRadius;
      result.insert(math::float3{v.x - r, v.y - r, v.z - r});
      result.insert(math::float3{v.x + r, v.y + r, v.z + r});
    }
  }
  return result;
}

// Curve //

Curve::Curve(BarneyGlobalState *s)
    : Geometry(s), m_index(this), m_vertexPosition(this), m_vertexRadius(this)
{}

void Curve::commitParameters()
{
  Geometry::commitParameters();

  m_index = getParamObject<Array1D>("primitive.index");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  m_vertexRadius = getParamObject<Array1D>("vertex.radius");

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on curve geometry");
    return;
  }
}

void Curve::finalize()
{
  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on curve geometry");
    return;
  }
}

void Curve::setBarneyParameters(BNGeom geom, BNContext context)
{
  int slot = 0;
  assert(m_vertexRadius->totalSize() == m_vertexPosition->totalSize());
  int numVertices =
      (int)std::min(m_vertexRadius->totalSize(), m_vertexPosition->totalSize());
  const bn_float3 *in_vertex = (const bn_float3 *)m_vertexPosition->data();
  const float *in_radius = (const float *)m_vertexRadius->data();
  std::vector<math::float4> vertex(numVertices);
  for (int i = 0; i < numVertices; i++)
    vertex[i] = math::float4(
        in_vertex[i].x, in_vertex[i].y, in_vertex[i].z, in_radius[i]);

  BNData vertices =
      bnDataCreate(context, slot, BN_FLOAT4, numVertices, vertex.data());
  bnSetData(geom, "vertices", vertices);

  int numIndices = (int)m_index->totalSize();
  std::vector<math::int2> index(numIndices);
  const int *in_index = (const int *)m_index->data();
  for (int i = 0; i < numIndices; i++) {
    index[i] = math::int2(in_index[i], in_index[i] + 1);
  }

#if 0
  // dump geometry to create test cases:
  std::cout << "std::vector<vec4f> vertices = {" << std::endl;
  for (auto v : vertex)
    std::cout << "{"<<v.x<<","<<v.y<<","<<v.z<<","<<v.w<<"}," << std::endl;
  std::cout << "};" << std::endl;
  std::cout << "std::vector<vec2i> index = {" << std::endl;
  for (auto v : index)
    std::cout << "{"<<v.x<<","<<v.y<<"}," << std::endl;
  std::cout << "};" << std::endl;
#endif

  BNData indices = bnDataCreate(
      context, slot, BN_INT2, index.size(), (const int *)index.data());
  bnSetData(geom, "indices", indices);

  addAttribute(geom, context, m_vertexAttributes[0], "vertex.attribute0");
  addAttribute(geom, context, m_vertexAttributes[1], "vertex.attribute1");
  addAttribute(geom, context, m_vertexAttributes[2], "vertex.attribute2");
  addAttribute(geom, context, m_vertexAttributes[3], "vertex.attribute3");
  addAttribute(geom, context, m_vertexAttributes[4], "vertex.color");
}

bool Curve::isValid() const
{
  return m_vertexPosition;
}

const char *Curve::bnSubtype() const
{
  return "capsules";
}

box3 Curve::bounds() const
{
  if (!isValid())
    return {};

  box3 result;
  for (size_t i = 0; i < m_vertexPosition->totalSize(); ++i) {
    math::float3 v = *(m_vertexPosition->beginAs<math::float3>() + i);
    float r = *(m_vertexRadius->beginAs<float>() + i);
    result.insert(math::float3{v.x - r, v.y - r, v.z - r});
    result.insert(math::float3{v.x + r, v.y + r, v.z + r});
  }
  return result;
}

// Triangle //

Triangle::Triangle(BarneyGlobalState *s)
    : Geometry(s), m_index(this), m_vertexPosition(this), m_vertexNormal(this)
{}

void Triangle::commitParameters()
{
  Geometry::commitParameters();
  m_index = getParamObject<Array1D>("primitive.index");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  m_vertexNormal = getParamObject<Array1D>("vertex.normal");
  m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
  m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
  m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
  m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
  m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");
}

void Triangle::finalize()
{
  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on triangle geometry");
    return;
  }

  m_generatedIndices.clear();
  if (!m_index) {
    m_generatedIndices.resize(m_vertexPosition->totalSize());
    std::iota(m_generatedIndices.begin(), m_generatedIndices.end(), 0);
  }
}

bool Triangle::isValid() const
{
  return m_vertexPosition;
}

void Triangle::setBarneyParameters(BNGeom geom, BNContext context)
{
  int slot = 0;
  int numVertices = (int)m_vertexPosition->totalSize();
  int numIndices =
      m_index ? (int)m_index->size() : (int)(m_generatedIndices.size() / 3);
  const bn_float3 *vertices = (const bn_float3 *)m_vertexPosition->data();
  const bn_int3 *indices = m_index ? (const bn_int3 *)m_index->data()
                                   : (const bn_int3 *)m_generatedIndices.data();

  BNData _vertices =
      bnDataCreate(context, slot, BN_FLOAT3, numVertices, vertices);
  bnSetAndRelease(geom, "vertices", _vertices);

  BNData _indices = bnDataCreate(context, slot, BN_INT3, numIndices, indices);
  bnSetAndRelease(geom, "indices", _indices);

  if (m_vertexNormal) {
    const bn_float3 *normals = (const bn_float3 *)m_vertexNormal->data();
    BNData _normals =
        bnDataCreate(context, slot, BN_FLOAT3, numVertices, normals);
    bnSetAndRelease(geom, "normals", _normals);
  }

  addAttribute(geom, context, m_attributes[0], "primitive.attribute0");
  addAttribute(geom, context, m_attributes[1], "primitive.attribute1");
  addAttribute(geom, context, m_attributes[2], "primitive.attribute2");
  addAttribute(geom, context, m_attributes[3], "primitive.attribute3");
  addAttribute(geom, context, m_attributes[4], "primitive.color");

  addAttribute(geom, context, m_vertexAttributes[0], "vertex.attribute0");
  addAttribute(geom, context, m_vertexAttributes[1], "vertex.attribute1");
  addAttribute(geom, context, m_vertexAttributes[2], "vertex.attribute2");
  addAttribute(geom, context, m_vertexAttributes[3], "vertex.attribute3");
  addAttribute(geom, context, m_vertexAttributes[4], "vertex.color");

  bnCommit(geom);
}

const char *Triangle::bnSubtype() const
{
  return "triangles";
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

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Geometry *);
