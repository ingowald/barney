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
  if (m_barneyModel)
    bnRelease(m_barneyModel);
}

bool World::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  if (name == "bounds" && type == ANARI_FLOAT32_BOX3) {
    if (flags & ANARI_WAIT) {
      deviceState()->waitOnCurrentFrame();
      deviceState()->commitBufferFlush();
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

void World::commit()
{
  m_zeroSurfaceData = getParamObject<ObjectArray>("surface");
  m_zeroVolumeData = getParamObject<ObjectArray>("volume");
  m_zeroLightData = getParamObject<ObjectArray>("light");
  
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

  m_zeroGroup->commit();
  m_zeroInstance->commit();

  m_instanceData = getParamObject<ObjectArray>("instance");

  m_instances.clear();
  if (m_instanceData) {
    m_instanceData->removeAppendedHandles();
    if (addZeroInstance)
      m_instanceData->appendHandle(m_zeroInstance.ptr);
    std::for_each(m_instanceData->handlesBegin(),
        m_instanceData->handlesEnd(),
        [&](auto *o) {
          if (o && o->isValid())
            m_instances.push_back((Instance *)o);
        });
  } else if (addZeroInstance)
    m_instances.push_back(m_zeroInstance.ptr);
}

BNModel World::makeCurrent()
{
  auto *state = deviceState();

  if (deviceState()->currentWorld != this) {
    if (m_barneyModel)
      bnRelease(m_barneyModel);
    m_barneyModel = nullptr;
    m_lastBarneyModelBuild = 0;
    m_barneyModel = bnModelCreate(state->context);
    state->currentWorld = this;
  }

  if (state->objectUpdates.lastSceneChange > m_lastBarneyModelBuild)
    buildBarneyModel();

  return m_barneyModel;
}

void World::buildBarneyModel()
{
  reportMessage(ANARI_SEVERITY_DEBUG, "barney::World rebuilding model");

  std::vector<const Group *> groups;
  std::vector<BNGroup> barneyGroups;
  std::vector<BNTransform> barneyTransforms;

  int numGPUs = 4;
  size_t numGroupsLocal = m_instances.size();
  size_t numGroupsTotal = m_instances.size() * numGPUs;

  groups.reserve(numGroupsTotal);
  barneyGroups.resize(numGroupsTotal, nullptr);
  barneyTransforms.reserve(numGroupsTotal);

  for (auto inst : m_instances) {
    barneyTransforms.push_back(*inst->barneyTransform());
    groups.push_back(inst->group());
  }

  if (barneyTransforms.size() != numGroupsLocal) {
    reportMessage(ANARI_SEVERITY_FATAL_ERROR,
        "Barney transforms and groups are different sizes!");
    return;
  }

  for (int gpuID=0;gpuID<numGPUs;gpuID++) {
    for (size_t i = 0; i < numGroupsLocal; i++) {
      int index = numGroupsLocal * numGPUs + i;
      if (barneyGroups[index] != nullptr)
        continue;
      auto *g = groups[i];
      BNGroup bg = g->makeBarneyGroup(m_barneyModel, gpuID);
      for (size_t j = i; j < numGroupsLocal; j++) {
        int jndex = numGroupsLocal * numGPUs + j;
        if (groups[j] == g)
          barneyGroups[jndex] = bg;
      }
    }

    bnSetInstances(m_barneyModel,
        gpuID,
        barneyGroups.data() + gpuID * numGroupsLocal,
        barneyTransforms.data() + gpuID * numGroupsLocal,
        numGroupsLocal);
    bnBuild(m_barneyModel, gpuID);
  }

  std::set<BNGroup> uniqueBarneyGroups;
  for (auto bng : barneyGroups)
    uniqueBarneyGroups.insert(bng);
  for (auto bng : uniqueBarneyGroups)
    bnRelease(bng);

  m_lastBarneyModelBuild = helium::newTimeStamp();
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::World *);
