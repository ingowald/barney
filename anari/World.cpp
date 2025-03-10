// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "World.h"
// std
#include <algorithm>
#include <set>

namespace barney_device {

World::World(BarneyGlobalState *s)
    : Object(ANARI_WORLD, s),
      m_zeroSurfaceData(this),
      m_zeroVolumeData(this),
      m_zeroLightData(this),
      m_instanceData(this)
{
  m_zeroGroup = new Group(s);
  m_zeroInstance = new Instance(s);
  m_zeroInstance->setParamDirect("group", m_zeroGroup.ptr);

  // never any public ref to these objects
  m_zeroGroup->refDec(helium::RefType::PUBLIC);
  m_zeroInstance->refDec(helium::RefType::PUBLIC);
}

World::~World()
{
  if (m_barneyModel) {
    bnRelease(m_barneyModel);
    m_barneyModel = 0;
  }
  auto *state = deviceState();
  if (state->currentWorld   == this)
    state->currentWorld = nullptr;
}

bool World::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  if (name == "bounds" && type == ANARI_FLOAT32_BOX3) {
    if (flags & ANARI_WAIT) {
      deviceState()->commitBuffer.flush();
      makeCurrent();
    }
    box3 bounds;
    bounds.invalidate();
    std::for_each(m_instances.begin(), m_instances.end(), [&](auto *inst) {
      bounds.insert(inst->bounds());
    });
    std::memcpy(ptr, &bounds, sizeof(bounds));
    return true;
  }

  return Object::getProperty(name, type, ptr, flags);
}

void World::commitParameters()
{
  m_zeroSurfaceData = getParamObject<ObjectArray>("surface");
  m_zeroVolumeData = getParamObject<ObjectArray>("volume");
  m_zeroLightData = getParamObject<ObjectArray>("light");
  m_instanceData = getParamObject<ObjectArray>("instance");
}

void World::finalize()
{
  const bool addZeroInstance =
      m_zeroSurfaceData || m_zeroVolumeData || m_zeroLightData;
  if (addZeroInstance)
    reportMessage(ANARI_SEVERITY_DEBUG, "barney::World will add zero instance");

  if (m_zeroSurfaceData) {
    reportMessage(ANARI_SEVERITY_DEBUG,
        "barney::World found %zu surfaces in zero instance",
        m_zeroSurfaceData->size());
    m_zeroGroup->setParamDirect("surface", getParamDirect("surface"));
  } else
    m_zeroGroup->removeParam("surface");

  if (m_zeroVolumeData) {
    reportMessage(ANARI_SEVERITY_DEBUG,
        "barney::World found %zu volumes in zero instance",
        m_zeroVolumeData->size());
    m_zeroGroup->setParamDirect("volume", getParamDirect("volume"));
  } else
    m_zeroGroup->removeParam("volume");

  if (m_zeroLightData) {
    reportMessage(ANARI_SEVERITY_DEBUG,
        "barney::World found %zu lights in zero instance",
        m_zeroLightData->size());
    m_zeroGroup->setParamDirect("light", getParamDirect("light"));
  } else
    m_zeroGroup->removeParam("light");

  m_zeroInstance->setParam("id", getParam<uint32_t>("id", ~0u));

  m_zeroGroup->commitParameters();
  m_zeroInstance->commitParameters();
  m_zeroGroup->finalize();
  m_zeroInstance->finalize();

  m_instances.clear();

  if (m_instanceData) {
    std::for_each(m_instanceData->handlesBegin(),
        m_instanceData->handlesEnd(),
        [&](auto *o) {
          if (o && o->isValid())
            m_instances.push_back((Instance *)o);
        });
  }

  if (addZeroInstance)
    m_instances.push_back(m_zeroInstance.ptr);
}

BNModel World::makeCurrent()
{
  auto *state = deviceState();
  
  if (state->currentWorld != this) {
    if (m_barneyModel)
      bnRelease(m_barneyModel);
    m_barneyModel = nullptr;
    m_lastBarneyModelBuild = 0;
    m_barneyModel = bnModelCreate(state->context);
    assert(m_barneyModel);
    state->currentWorld = this;
  }

  assert(m_barneyModel);
  if (state->objectUpdates.lastSceneChange > m_lastBarneyModelBuild)
    buildBarneyModel();

  assert(m_barneyModel);
  return m_barneyModel;
}

void World::buildBarneyModel()
{
  reportMessage(ANARI_SEVERITY_DEBUG, "barney::World rebuilding model");

  std::vector<const Group *> groups;
  std::vector<BNGroup> barneyGroups;
  std::vector<BNTransform> barneyTransforms;

  groups.reserve(m_instances.size());
  barneyGroups.resize(m_instances.size(), nullptr);
  barneyTransforms.reserve(m_instances.size());

  for (auto inst : m_instances) {
    barneyTransforms.push_back(*inst->barneyTransform());
    groups.push_back(inst->group());
  }

  
  for (size_t i = 0; i < groups.size(); i++) {
    if (barneyGroups[i] != nullptr)
      continue;
    auto *g = groups[i];
    BNGroup bg = g->makeBarneyGroup(getContext());
    for (size_t j = i; j < groups.size(); j++) {
      if (groups[j] == g)
        barneyGroups[j] = bg;
    }
  }
  
  if (barneyTransforms.size() != barneyGroups.size()) {
    reportMessage(ANARI_SEVERITY_FATAL_ERROR,
        "Barney transforms and groups are different sizes!");
    return;
  }

  assert(m_barneyModel);
  bnSetInstances(m_barneyModel,
      0,
      barneyGroups.data(),
      barneyTransforms.data(),
      (int)barneyGroups.size());
  bnBuild(m_barneyModel, 0);

  std::set<BNGroup> uniqueBarneyGroups;
  for (auto bng : barneyGroups)
    uniqueBarneyGroups.insert(bng);
  for (auto bng : uniqueBarneyGroups)
    bnRelease(bng);

  m_lastBarneyModelBuild = helium::newTimeStamp();
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::World *);
