#include "rtcore/common/Backend.h"
#ifdef _WIN32
# include <windows.h>
#else
# include <dlfcn.h>
#endif


namespace barney {
  namespace rtc {
#if !BARNEY_BACKEND_CUDA
    Backend *createBackend_cuda()
    {
      throw std::runtime_error("#barney: cuda support not compiled in!");
    }
#endif
#if !BARNEY_BACKEND_OPTIX
    Backend *createBackend_optix()
    {
      throw std::runtime_error("#barney: optix support not compiled in!");
    }
#endif
#if !BARNEY_BACKEND_EMBREE
    Backend *createBackend_embree()
    {
      throw std::runtime_error("#barney: embree/cpu support not compiled in!");
    }
#endif

    int Backend::getDeviceCount()
    {
      return get()->numPhysicalDevices;
    }
  
    Backend *Backend::create()
    {
#if 1
      return createBackend_optix();
#else
      const char *_fromEnv = getenv("BARNEY_BACKEND");;
      std::string fromEnv = _fromEnv?_fromEnv:"";
      if (fromEnv == "") {
        // nothing to do, leave defautl selection below
      } else if (fromEnv == "optix") {
        PING;
        return createBackend_optix();
      } else if (fromEnv == "embree" || fromEnv == "cpu") {
        return createBackend_embree();
      } else if (fromEnv == "cuda") {
        return createBackend_cuda();
      } else {
        throw std::runtime_error
          ("#barney: user requested unknown barney backend '"+fromEnv+"'");
      }

      Backend *be = nullptr;
#if BARNEY_HAVE_CUDA
      try {
        be = createBackend_optix();
      } catch (std::exception &e) { PING; PRINT(e.what()); }
      if (!be) {
        try {
          be = createBackend_cuda();
        } catch (std::exception &e) {PING; PRINT(e.what()); }
      }
      if (be && be->numPhysicalDevices == 0) {
        std::cout << "#barney: no GPUs found; trying CPU callback" << std::endl;
        delete be;
        be = nullptr;
      }
#endif
      be = createBackend_embree();
      return be;
#endif
    }
  
    Backend *Backend::get()
    {
      if (!singleton) singleton = create();
      PING; PRINT(singleton);
      return singleton;
    }

    const void *getSymbol(const std::string &symbolName)
    {
#ifdef _WIN32
      HMODULE libCurrent = GetModuleHandle(NULL);
      void* symbol = (void*)GetProcAddress(libCurrent, symbolName.c_str());
#else
      void *lib = dlopen(nullptr,RTLD_GLOBAL);
      void* symbol = (void*)dlsym(lib, symbolName.c_str());
#endif
      if (!symbol)
        throw std::runtime_error
          ("#barney::rtcore: could not find required symbol "
           +symbolName);
      return symbol;
    }
    
    Backend *Backend::singleton = nullptr;
  }
}



