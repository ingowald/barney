// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Material.h"
#include "common.h"
// CUDA
#include <vector_functions.h>

namespace barney_device {

Material::Material(BarneyGlobalState *s) : Object(ANARI_MATERIAL, s) {}

Material::~Material() = default;

Material *Material::createInstance(
    std::string_view subtype, BarneyGlobalState *s)
{
  if (subtype == "matte")
    return new Matte(s);
  else if (subtype == "physicallyBased")
    return new PhysicallyBased(s);
  else
    return (Material *)new UnknownObject(ANARI_MATERIAL, s);
}

void Material::markCommitted()
{
  deviceState()->markSceneChanged();
  Object::markCommitted();
}

// Subtypes ///////////////////////////////////////////////////////////////////

// Matte //

Matte::Matte(BarneyGlobalState *s) : Material(s) {}

void Matte::commit()
{
  Object::commit();

  m_color = math::float4(1.f, 1.f, 1.f, 1.f);
  getParam("color", ANARI_FLOAT32_VEC3, &m_color);
  getParam("color", ANARI_FLOAT32_VEC4, &m_color);
  m_colorSampler = getParamObject<Sampler>("color");
}

BNMaterial Matte::makeBarneyMaterial(BNModel model, int slot) const
{
  BNMaterial mat = bnMaterialCreate(model, slot, "matte");
  bnSet3f(mat, "reflectance", m_color.x, m_color.y, m_color.z);
#if 1 // Hack to get samplers working on Matte material
  if (m_colorSampler) {
    if (auto imgSampler = dynamic_cast<const Image1D *>(m_colorSampler.ptr)) {
      BNData imageData = makeBarneyData(model, slot, imgSampler->m_image);

      if (imageData) {
        bnSetString(mat, "sampler.type", "image1D");
        bnSet1i(mat, "sampler.image1D.inAttribute", imgSampler->m_inAttribute);
        bnSet4x4fv(mat, "sampler.image1D.inTransform", (const float *)&imgSampler->m_inTransform.x);
        bnSet4f(mat, "sampler.image1D.inOffset",
                imgSampler->m_inOffset.x,
                imgSampler->m_inOffset.y,
                imgSampler->m_inOffset.z,
                imgSampler->m_inOffset.w);
        bnSet4x4fv(mat, "sampler.image1D.outTransform", (const float *)&imgSampler->m_outTransform.x);
        bnSet4f(mat, "sampler.image1D.outOffset",
                imgSampler->m_outOffset.x,
                imgSampler->m_outOffset.y,
                imgSampler->m_outOffset.z,
                imgSampler->m_outOffset.w);

        bnSetData(mat, "sampler.image1D.image.data", imageData);
        bnSet1i(mat, "sampler.image1D.image.width", imgSampler->m_image->size());
        // TODO: wrapMode, filterMode!!
      }
    }
    if (auto imgSampler = dynamic_cast<const Image2D *>(m_colorSampler.ptr)) {
      BNData imageData = makeBarneyData(model, slot, imgSampler->m_image);

      if (imageData) {
        bnSetString(mat, "sampler.type", "image2D");
        bnSet1i(mat, "sampler.image2D.inAttribute", imgSampler->m_inAttribute);
        bnSet4x4fv(mat, "sampler.image2D.inTransform", (const float *)&imgSampler->m_inTransform.x);
        bnSet4f(mat, "sampler.image2D.inOffset",
                imgSampler->m_inOffset.x,
                imgSampler->m_inOffset.y,
                imgSampler->m_inOffset.z,
                imgSampler->m_inOffset.w);
        bnSet4x4fv(mat, "sampler.image2D.outTransform", (const float *)&imgSampler->m_outTransform.x);
        bnSet4f(mat, "sampler.image2D.outOffset",
                imgSampler->m_outOffset.x,
                imgSampler->m_outOffset.y,
                imgSampler->m_outOffset.z,
                imgSampler->m_outOffset.w);

        bnSetData(mat, "sampler.image2D.image.data", imageData);
        bnSet1i(mat, "sampler.image2D.image.width", imgSampler->m_image->size().x);
        bnSet1i(mat, "sampler.image2D.image.height", imgSampler->m_image->size().y);
        // TODO: wrapMode, filterMode!!
      }
    }
    else if (auto xfmSampler = dynamic_cast<const TransformSampler *>(m_colorSampler.ptr)) {
      bnSetString(mat, "sampler.type", "transform");
      bnSet1i(mat, "sampler.transform.inAttribute", xfmSampler->m_inAttribute);
      bnSet4x4fv(mat, "sampler.transform.outTransform",
                 (const float *)&xfmSampler->m_outTransform.x);
      bnSet4f(mat, "sampler.transform.outOffset",
              xfmSampler->m_outOffset.x,
              xfmSampler->m_outOffset.y,
              xfmSampler->m_outOffset.z,
              xfmSampler->m_outOffset.w);
    }
  }
#endif
  return mat;
}

// PhysicallyBased //

PhysicallyBased::PhysicallyBased(BarneyGlobalState *s) : Material(s) {}

void PhysicallyBased::commit()
{
  Object::commit();

  m_baseColor.value = math::float4(1.f, 1.f, 1.f, 1.f);
  getParam("baseColor", ANARI_FLOAT32_VEC3, &m_baseColor.value);
  getParam("baseColor", ANARI_FLOAT32_VEC4, &m_baseColor.value);

  m_emissive.value = math::float3(0.f, 0.f, 0.f);
  getParam("emissive", ANARI_FLOAT32_VEC3, &m_emissive.value);

  m_specularColor.value = math::float3(1.f, 1.f, 1.f);
  getParam("specularColor", ANARI_FLOAT32_VEC3, &m_specularColor.value);

  m_opacity.value = 1.f;
  getParam("opacity", ANARI_FLOAT32, &m_opacity.value);

  m_metallic.value = 1.f;
  getParam("metallic", ANARI_FLOAT32, &m_metallic.value);

  m_roughness.value = 1.f;
  getParam("roughness", ANARI_FLOAT32, &m_roughness.value);

  m_specular.value = 0.f;
  getParam("specular", ANARI_FLOAT32, &m_specular.value);

  m_transmission.value = 0.f;
  getParam("transmission", ANARI_FLOAT32, &m_transmission.value);

  m_ior = 1.5f;
  getParam("ior", ANARI_FLOAT32, &m_ior);
}

BNMaterial PhysicallyBased::makeBarneyMaterial(BNModel model, int slot) const
{
  BNMaterial mat = bnMaterialCreate(model, slot, "physicallyBased");

  bnSet3f(mat, "baseColor",
      m_baseColor.value.x, m_baseColor.value.y, m_baseColor.value.z);

  bnSet3f(mat, "emissive",
      m_emissive.value.x, m_emissive.value.y, m_emissive.value.z);

  bnSet3f(mat, "specularColor",
      m_specularColor.value.x, m_specularColor.value.y, m_specularColor.value.z);

  bnSet1f(mat, "opacity", m_opacity.value);
  bnSet1f(mat, "metallic", m_metallic.value);
  bnSet1f(mat, "roughness", m_roughness.value);
  bnSet1f(mat, "specular", m_specular.value);
  bnSet1f(mat, "transmission", m_transmission.value);
  bnSet1f(mat, "ior", m_ior);

  return mat;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Material *);
