#pragma once

#include <stdint.h>
#include <string>
#include <cuda.h>

namespace barney {
  namespace logging {
    typedef int64_t Event;
    enum class EventType { CPU, GPU, };
    Event newEvent(EventType et = EventType::CPU, CUstream stream = 0, int gpuID = 0);
    void enqueueEvent(Event ev);
    bool pollEvent(Event ev);
    void logEvent(Event ev, std::string str);
    void printEventLog(bool clear=true);
  } // ::barney::logging
} // ::barney

