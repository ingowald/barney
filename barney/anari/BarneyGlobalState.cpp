// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define ANARI_BARNEY_MATH_DEFINITIONS 1

#include "anari/BarneyGlobalState.h"
#include "anari/Frame.h"
#include "anari/common.h"

// for dlsym
#ifdef _WIN32
# include <windows.h>
#else
# include <dlfcn.h>
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

namespace BARNEY_NS {
  namespace anari {
    
    Tether::~Tether()
    {
      BANARI_TRACK_LEAKS(std::cout << "#banari: tether destructing - "
                         "releasing barney context" << std::endl);
      if (context) { bnContextDestroy(context); context = 0; }
    }

    BarneyGlobalState::BarneyGlobalState(ANARIDevice d)
      : helium::BaseGlobalDeviceState(d)
    {}

    BarneyGlobalState::~BarneyGlobalState()
    {
      BANARI_TRACK_LEAKS(std::cout << "#banari: barneyglobalstate destructing"
                         " - releasing tether" << std::endl);
    }
  
    void BarneyGlobalState::markSceneChanged()
    {
      objectUpdates.lastSceneChange = helium::newTimeStamp();
    }

    void BarneyGlobalState::markStructuralSceneChanged()
    {
      auto ts = helium::newTimeStamp();
      objectUpdates.lastSceneChange = ts;
      objectUpdates.lastStructuralChange = ts;
    }

    bool Tether::allDevicesPresent()
    {
      for (auto dev : devices)
        if (dev == 0) return false;
      return true;
    }

    TetheredModel::SP Tether::getOrCreateTetheredModel(int uniqueID)
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (activeModels.find(uniqueID) != activeModels.end()) {
        BANARI_TRACK_LEAKS(std::cout << "#banari returning already created model "
                           << uniqueID << std::endl);
        return activeModels[uniqueID]->shared_from_this();
      }

      BANARI_TRACK_LEAKS(std::cout << "#banari creating new tethered model "
                         << uniqueID << std::endl);
      TetheredModel::SP newModel = std::make_shared<TetheredModel>(this,uniqueID);
      return newModel;
    }

    TetheredModel::TetheredModel(Tether *tether, int uniqueID)
      : tether(tether),
        uniqueID(uniqueID)
    {
      BANARI_TRACK_LEAKS(std::cout << "#banari: creating new tetherd model ID "
                         << uniqueID << std::endl);
      model = bnModelCreate(tether->context);
      BANARI_TRACK_LEAKS(std::cout << "#banari: created new barney model "
                         << (int*)model << std::endl);
    
      // iw do NOT try to lock tether, it's already locked when it creates us!
      tether->activeModels[uniqueID] = this;
    }
  
    TetheredModel::~TetheredModel()
    {
      BANARI_TRACK_LEAKS(std::cout << "#banari: tethered model is dying" << std::endl);
      std::lock_guard<std::mutex> lock(tether->mutex);
      tether->activeModels.erase(tether->activeModels.find(uniqueID));
    
      if (model) {
        BANARI_TRACK_LEAKS(std::cout << "#banari: releasing barney model handle ID "
                           << uniqueID << std::endl);
        bnRelease(model);
      }
    }

    // ==================================================================
    // plugin infrastructure - should at some point more to a separate
    // compilation unit
    // ==================================================================

    /*! expects a string of the form '<geomtype>@<pluginname>'. may
      return null if either the plugin cannot be found (ie, it's not
      included in the build), or that plugin for some reason cannot
      create that geometry. */
    Geometry *PluginInfrastructure::newGeometry(const std::string_view &typeAtPlugin,
                                                BarneyGlobalState *banari)
    {
      std::string realName;
      Plugin *plugin = findPlugin(typeAtPlugin,realName);
      if (!plugin) return nullptr;

      auto creatorFromPlugin = plugin->supportedGeometries[realName];
      if (!creatorFromPlugin) return nullptr;

      return creatorFromPlugin(banari);
    }

    /*! expects a string of the form '<geomtype>@<pluginname>'. may
      return null if either the plugin cannot be found (ie, it's not
      included in the build), or that plugin for some reason cannot
      create that geometry. */
    SpatialField *PluginInfrastructure::newSpatialField(const std::string_view &typeAtPlugin,
                                                BarneyGlobalState *banari)
    {
      std::string realName;
      Plugin *plugin = findPlugin(typeAtPlugin,realName);
      if (!plugin) return nullptr;

      auto creatorFromPlugin = plugin->supportedSpatialFields[realName];
      if (!creatorFromPlugin) return nullptr;

      return creatorFromPlugin(banari);
    }

    /*! takes a "<providedType>@<registeredPlugin>" string, splits it
        into its two components, looks up the respective plugin, and
        either returns already loaed plugin or tries to load it if
        required. If all went well this returns the fully loaded
        plugin, as well as the properly stripped 'providedType'
        string; if anything went wrong (not a valid plugin object
        string or plugin not found and could not be loaded) this
        returns nullptr */
    PluginInfrastructure::Plugin *
    PluginInfrastructure::findPlugin(const std::string_view &typeAtPlugin,
                                     std::string &requestedType)
    {
      int pos = typeAtPlugin.find("@");
      if (pos == typeAtPlugin.npos)
        return nullptr;

      const std::string pluginName = std::string(typeAtPlugin.substr(pos+1));
      requestedType = typeAtPlugin.substr(0,pos);

      if (alreadyLoadedPlugins.find(pluginName) == alreadyLoadedPlugins.end()) 
        alreadyLoadedPlugins[pluginName] = loadPlugin(pluginName);

      Plugin *plugin = alreadyLoadedPlugins[pluginName];
      // may still be null if plugin loading failed:
      return plugin;
    }

    PluginInfrastructure::Plugin *
    PluginInfrastructure::loadPlugin(const std::string_view &pluginName)
    {
      Plugin *plugin = nullptr;
      const std::string symbolName
        = std::string("registerPlugin_barney_")
        + std::string(TOSTRING(BARNEY_BACKEND_NAME))
        + "_"
        + std::string(pluginName);

      PluginInfrastructure::Plugin *(*registerPlugin)() = 0;
#ifdef _WIN32
      auto module = GetModuleHandle(NULL);
      registerPlugin
        = (PluginInfrastructure::Plugin *(*)())
        GetProcAddress(hSelf, symbolName.c_str());
#else
      registerPlugin
        = (PluginInfrastructure::Plugin *(*)())
        dlsym(nullptr,symbolName.c_str());
#endif
      if (!registerPlugin) return nullptr;
      return registerPlugin();
    }

        // called by plugin registry function to declare a new geometry type */
    void PluginInfrastructure::Plugin
    ::exportGeometry(const std::string &typeName,
                     Geometry*(*creatorFunction)(BarneyGlobalState*))
    {
      supportedGeometries[typeName] = creatorFunction;
    }
    
    // called by plugin registry function to declare a new spatial field type */
    void PluginInfrastructure::Plugin
    ::exportSpatialField(const std::string &typeName,
                         SpatialField*(*creatorFunction)(BarneyGlobalState*))
    {
      supportedSpatialFields[typeName] = creatorFunction;
    }
    
  }
}
