// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Surface.h"

namespace tally_device {

  Surface::Surface(TallyGlobalState *s) : Object(ANARI_SURFACE, s) {}

  Surface::~Surface() = default;

  void Surface::commit()
  {
    m_id = getParam<uint32_t>("id", ~0u);
    m_geometry = getParamObject<Geometry>("geometry");
    m_material = getParamObject<Material>("material");

    if (!m_material) {
      reportMessage(ANARI_SEVERITY_WARNING, "missing 'material' on ANARISurface");
      return;
    }

    if (!m_geometry) {
      reportMessage(ANARI_SEVERITY_WARNING, "missing 'geometry' on ANARISurface");
      return;
    }

    setTallyParameters();
  }

  void Surface::markCommitted()
  {
    deviceState()->markSceneChanged();
    Object::markCommitted();
  }

  const Geometry *Surface::geometry() const
  {
    return m_geometry.ptr;
  }

  const Material *Surface::material() const
  {
    return m_material.ptr;
  }

  TallyGeom::SP Surface::getTallyGeom(TallyModel::SP model, int slot)
  {
    if (!isModelTracked(model, slot)) {
      cleanup();
      trackModel(model, slot);
      m_bnGeom
        // = bnGeometryCreate(model, slot, m_geometry->bnSubtype());
        = TallyGeom::create(m_geometry->bnSubtype(),this);
      setTallyParameters();
    }

    return m_bnGeom;
  }

  bool Surface::isValid() const
  {
    return m_geometry && m_material && m_geometry->isValid()
      && m_material->isValid();
  }

  void Surface::setTallyParameters()
  {
    if (!isValid() || !m_bnGeom)
      return;
    TallyModel::SP model = trackedModel();
    int slot = trackedSlot();
    // bnSetObject(m_bnGeom, "material", m_material->getTallyMaterial(model, slot));
    if(m_bnGeom) m_bnGeom->material = m_material->getTallyMaterial(model, slot);
    m_geometry->setTallyParameters(m_bnGeom, model, slot);
    // bnCommit(m_bnGeom);
  }

  void Surface::cleanup()
  {
    // if (m_bnGeom)
    //   bnRelease(m_bnGeom);
    m_bnGeom = nullptr;
  }

} // namespace tally_device

TALLY_ANARI_TYPEFOR_DEFINITION(tally_device::Surface *);
