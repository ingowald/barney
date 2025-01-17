#include "rtcore/common/Backend.h"

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
      const char *_fromEnv = getenv("BARNEY_BACKEND");;
      std::string fromEnv = _fromEnv?_fromEnv:"";
      if (fromEnv == "") {
        // nothing to do, leave defautl selection below
      } else if (fromEnv == "optix") {
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
      } catch (...) {}
      if (!be) {
        try {
          be = createBackend_cuda();
        } catch (...) {}
      }
      if (be && be->numPhysicalDevices == 0) {
        std::cout << "#barney: no GPUs found; trying CPU callback" << std::endl;
        delete be;
        be = nullptr;
      }
#endif
      be = createBackend_embree();
      return be;
    }
  
    Backend *Backend::get()
    {
      if (!singleton) singleton = create();
      return singleton;
    }
  
    Backend *Backend::singleton = nullptr;
  }
}



