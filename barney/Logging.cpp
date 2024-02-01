#include <algorithm>
#include <iostream>
#include <mutex>
#include <ostream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>
#include "Logging.h"

#ifdef LOGGING_USE_MPI
  #include <mpi.h>
#endif

#ifdef LOGGING_USE_OMP
  #include <omp.h>
  #define OMP_TIME_TO_MILLISECONDS(t) (((t) * 1000.0))
#else
  #include <chrono>  
#endif

#ifndef NDEBUG
#define CUDA_SAFE_CALL(FUNC) { cuda_safe_call((FUNC), __FILE__, __LINE__); }
#else
#define CUDA_SAFE_CALL(FUNC) FUNC
#endif

inline void cuda_safe_call(cudaError_t code, char const* file, int line)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s %s:%i\n", cudaGetErrorString(code), file, line);
  }
}

namespace barney {
  namespace logging {

#ifdef LOGGING_USE_OMP
  double global_time0 = 0;
  double local_time0 = 0;
  void setFirstTime(double local_time, double global_time) {
      local_time0 = local_time;
      global_time0 = global_time;
  }
#else
    using Clock = std::chrono::steady_clock;
#endif

    struct EventImpl {
      Event handle;
      EventType type;
      struct {
#ifdef LOGGING_USE_OMP
        double start, stop;
#else        
        std::chrono::time_point<Clock> start, stop;
#endif        
      } onCPU;
      struct {
        cudaEvent_t start, stop;
        CUstream stream;
        int gpuID;
      } onGPU;
      bool isAsync;
      bool complete;
    };

    struct EventRec {
      EventRec() = default;
      EventRec(Event ev, 
#ifdef LOGGING_USE_OMP      
        double t, 
#else
        std::chrono::milliseconds t, 
#endif        
        std::string s, double dur, int gpuID)
        : event(ev), time(t), str(s), duration(dur), gpuID(gpuID)  {}
      Event event;
#ifdef LOGGING_USE_OMP
      double time;
#else      
      std::chrono::milliseconds time;
#endif      
      std::string str;
      double duration = 0.;
      int gpuID = -1;
    };

    static std::mutex mtx;
    static std::vector<EventImpl> events;
    static std::vector<EventRec> eventLog;

    namespace state {
      static bool first = true;
#ifdef LOGGING_USE_OMP
      //static double t0;
#else      
      static std::chrono::time_point<Clock> t0;
#endif      
    }

    Event newEvent(EventType et, CUstream stream, int gpuID)
    {
      if (state::first) {
#ifdef LOGGING_USE_OMP
        //state::t0 = omp_get_wtime();
#else        
        state::t0 = Clock::now();
#endif        
        state::first = false;
      }

      static Event ev = 0;

      Event handle = ev++;

      EventImpl impl;
      impl.handle = handle;
      impl.type = et;
      if (et == EventType::GPU) {
        int prevID;
        CUDA_SAFE_CALL(cudaGetDevice(&prevID));
        CUDA_SAFE_CALL(cudaSetDevice(gpuID));
        CUDA_SAFE_CALL(cudaEventCreate(&impl.onGPU.start));
        CUDA_SAFE_CALL(cudaEventCreate(&impl.onGPU.stop));
        impl.onGPU.stream = stream;
        impl.onGPU.gpuID = gpuID;
        CUDA_SAFE_CALL(cudaEventRecord(impl.onGPU.start,impl.onGPU.stream));
        CUDA_SAFE_CALL(cudaSetDevice(prevID));
      } else {
#ifdef LOGGING_USE_OMP
        impl.onCPU.start = OMP_TIME_TO_MILLISECONDS(omp_get_wtime());
#else        
        impl.onCPU.start = Clock::now();
#endif        
      }
      impl.isAsync = false;
      impl.complete = false;

      std::unique_lock<std::mutex> lck(mtx);
      events.push_back(impl);
      lck.unlock();

      return handle;
    }

    void enqueueEvent(Event ev)
    {
      auto it = std::find_if(events.begin(),events.end(),[ev](EventImpl impl) { return impl.handle == ev; });
      if (it == events.end()) {
        std::cerr << "No such event: " << ev << '\n';
        return;
      }

      if (it->type == EventType::GPU) {
        it->isAsync = true;
        int prevID;
        CUDA_SAFE_CALL(cudaGetDevice(&prevID));
        CUDA_SAFE_CALL(cudaSetDevice(it->onGPU.gpuID));
        CUDA_SAFE_CALL(cudaEventRecord(it->onGPU.stop,it->onGPU.stream));
        CUDA_SAFE_CALL(cudaSetDevice(prevID));
      }
    }

    bool pollEvent(Event ev)
    {
      auto it = std::find_if(events.begin(),events.end(),[ev](EventImpl impl) { return impl.handle == ev; });
      if (it == events.end()) {
        std::cerr << "No such event: " << ev << '\n';
        return false;
      }

      if (it->complete)
        return false;

      if (it->type == EventType::GPU) {
        cudaError_t err = cudaEventQuery(it->onGPU.stop);
        if (err==cudaSuccess)
          it->complete = true;
        return err==cudaSuccess;
      }

      return false;
    }

    void logEvent(Event ev, std::string str)
    {
      auto it = std::find_if(events.begin(),events.end(),[ev](EventImpl impl) { return impl.handle == ev; });
      if (it == events.end()) {
        std::cerr << "No such event: " << ev << '\n';
        return;
      }

      double duration = 0.;
      int gpuID = -1;
      if (it->type == EventType::GPU) {
        if (!it->isAsync)
          CUDA_SAFE_CALL(cudaEventRecord(it->onGPU.stop,it->onGPU.stream));
        CUDA_SAFE_CALL(cudaEventSynchronize(it->onGPU.stop));
        float ms = 0.f;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&ms,it->onGPU.start,it->onGPU.stop));
        duration = (double)ms;
        gpuID = it->onGPU.gpuID;
      } else {
#ifdef LOGGING_USE_OMP
        it->onCPU.stop = OMP_TIME_TO_MILLISECONDS(omp_get_wtime());
        duration = it->onCPU.stop - it->onCPU.start;
#else   
        it->onCPU.stop = Clock::now();     
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(it->onCPU.stop - it->onCPU.start).count();
#endif        
      }

#ifdef LOGGING_USE_OMP
      double t = OMP_TIME_TO_MILLISECONDS(omp_get_wtime() - local_time0 + (local_time0 - global_time0));
#else      
      std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-state::t0);
#endif      
      std::unique_lock<std::mutex> lck(mtx);
      eventLog.push_back(EventRec(ev,t,str,duration,gpuID));
      lck.unlock();
    }

    void printEventLog(bool clear)
    {
      std::unique_lock<std::mutex> lck(mtx);
      std::vector<EventRec> cpyLog(eventLog);
      if (clear)
        eventLog.clear();
      lck.unlock();
      for (auto ev : cpyLog) {
#ifdef LOGGING_USE_OMP
        double tstart = ev.time - ev.duration;
#else        
        std::chrono::milliseconds tstart = ev.time - std::chrono::milliseconds((int64_t)ev.duration);
#endif        
        std::stringstream stream;
#ifdef LOGGING_USE_OMP        
        stream << ev.event << ';' << ((int64_t)tstart) << ';' << ((int64_t)ev.time) << ";\"" << ev.str << "\";" << ev.duration;
#else
        stream << ev.event << ';' << tstart.count() << ';' << ev.time.count() << ";\"" << ev.str << "\";" << ev.duration;
#endif        
        if (ev.gpuID >= 0) {
          int world_rank = 0;
          int device_count = 0;
#ifdef LOGGING_USE_MPI          
          MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);        
          cudaGetDeviceCount(&device_count);         
#endif          
          stream << ";GPU " << (ev.gpuID + device_count * world_rank);
        }
        stream << '\n';
        std::cout << stream.str();
      }
    }
  } // ::barney::logging
} // ::barney
