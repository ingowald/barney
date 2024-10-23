// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "World.h"
// std
#include <algorithm>
#include <set>

namespace tally_device {

World::World(TallyGlobalState *s)
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
  // if (m_tallyModel)
  //   bnRelease(m_tallyModel);
  m_tallyModel = {};
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
    reportMessage(ANARI_SEVERITY_DEBUG, "tally::World will add zero instance");

  if (m_zeroSurfaceData) {
    reportMessage(ANARI_SEVERITY_DEBUG,
                  "tally::World found %zu surfaces in zero instance",
        m_zeroSurfaceData->size());
    m_zeroGroup->setParamDirect("surface", getParamDirect("surface"));
  } else
    m_zeroGroup->removeParam("surface");

  if (m_zeroVolumeData) {
    reportMessage(ANARI_SEVERITY_DEBUG,
        "tally::World found %zu volumes in zero instance",
        m_zeroVolumeData->size());
    m_zeroGroup->setParamDirect("volume", getParamDirect("volume"));
  } else
    m_zeroGroup->removeParam("volume");

  if (m_zeroLightData) {
    reportMessage(ANARI_SEVERITY_DEBUG,
        "tally::World found %zu lights in zero instance",
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

TallyModel::SP World::makeCurrent()
{
  auto *state = deviceState();

  if (deviceState()->currentWorld != this) {
    // if (m_tallyModel)
    //   bnRelease(m_tallyModel);
    m_tallyModel = nullptr;
    m_lastTallyModelBuild = 0;
    m_tallyModel = TallyModel::create();//bnModelCreate(state->context);
    state->currentWorld = this;
  }

  if (state->objectUpdates.lastSceneChange > m_lastTallyModelBuild)
    buildTallyModel();

  return m_tallyModel;
}

void World::buildTallyModel()
{
  PING;
  reportMessage(ANARI_SEVERITY_DEBUG, "tally::World rebuilding model");

  std::vector<const Group *> groups;
  std::vector<TallyGroup::SP> tallyGroups;
  std::vector<TallyTransform> tallyTransforms;

  groups.reserve(m_instances.size());
  tallyGroups.resize(m_instances.size(), nullptr);
  tallyTransforms.reserve(m_instances.size());

  for (auto inst : m_instances) {
    tallyTransforms.push_back(*inst->tallyTransform());
    groups.push_back(inst->group());
  }

  for (size_t i = 0; i < groups.size(); i++) {
    if (tallyGroups[i] != nullptr)
      continue;
    auto *g = groups[i];
    TallyGroup::SP//TallyGroup::SP
      bg = g->makeTallyGroup(m_tallyModel, 0);
    for (size_t j = i; j < groups.size(); j++) {
      if (groups[j] == g)
        tallyGroups[j] = bg;
    }
  }

  if (tallyTransforms.size() != tallyGroups.size()) {
    reportMessage(ANARI_SEVERITY_FATAL_ERROR,
        "Tally transforms and groups are different sizes!");
    return;
  }

  PING;
  m_tallyModel->setInstances(tallyGroups,tallyTransforms);
  // bnSetInstances(m_tallyModel,
  //     0,
  //     tallyGroups.data(),
  //     tallyTransforms.data(),
  //     tallyGroups.size());
  // bnBuild(m_tallyModel, 0);

  // std::set<TallyGroup::SP> uniqueTallyGroups;
  // for (auto bng : tallyGroups)
  //   uniqueTallyGroups.insert(bng);
  // for (auto bng : uniqueTallyGroups)
  //   bnRelease(bng);

  m_lastTallyModelBuild = helium::newTimeStamp();
}

} // namespace tally_device

TALLY_ANARI_TYPEFOR_DEFINITION(tally_device::World *);
