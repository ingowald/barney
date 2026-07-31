// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "barney/FromEnv.h"

namespace BARNEY_NS {
  namespace native {
    
    FromEnv::FromEnv()
    {
      const char *e = getenv("BARNEY_CONFIG");
      if (!e) return;
      std::vector<std::string> components;
      std::string es = e;
      while (true) {
        size_t p = es.find(":");
        if (p == es.npos) {
          components.push_back(es);
          break;
        }
        components.push_back(es.substr(0,p));
        es = es.substr(p+1);
      }
      std::map<std::string,std::string> keyValue;
      for (auto comp : components) {
        size_t p = comp.find("=");
        if (p == comp.npos) {
          keyValue[comp] = "";
        } else {
          keyValue[comp.substr(0,p)] = comp.substr(p+1);
        }
      }
      for (auto kv : keyValue) {
        const std::string key = kv.first;
        const std::string value = kv.second;
      
        std::cout << "#barney.config " << key << " = '" << value << "'" << std::endl;

        if (value == "on" || value == "ON" || value == "1")
          boolValues[key] = 1;
        else if (value == "off" || value == "OFF" || value == "0")
          boolValues[key] = 0;
      
        if (key == "LOG_QUEUES" || key == "log_queues")
          logQueues = true;
        else if (key == "SKIP_DENOISING")
          skipDenoising = true;
        else if (key == "LOG_CONFIG" || key == "log_config")
          logConfig = true;
        else if (key == "LOG_BACKEND")
          logBackend = true;
        else if (key == "LOG_TOPO" || key == "log_topo")
          logTopo = true;
        else
          std::cerr << "Warning: unknown or unrecognized BARNEY_CONFIG key '" << key << "'" << std::endl;
      }
    }
    const FromEnv *FromEnv::get()
    {
      static std::mutex mutex;
      std::lock_guard<std::mutex> lock(mutex);
      static FromEnv *singleton = 0;
      if (!singleton) singleton = new FromEnv;
      return singleton;
    }

  }
}
