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
	    return 0;
      //throw std::runtime_error("#barney: cuda support not compiled in!");
    }
#endif
#if !BARNEY_BACKEND_OPTIX
    Backend *createBackend_optix()
    {
	    return 0;
      //throw std::runtime_error("#barney: optix support not compiled in!");
    }
#endif
#if !BARNEY_BACKEND_EMBREE
    Backend *createBackend_embree()
    {
	    return 0;
      //throw std::runtime_error("#barney: embree/cpu support not compiled in!");
    }
#endif


    void Buffer::uploadAsync(const void *hostPtr,
                              size_t numBytes,
                              size_t ofs)
    {
      device->copyAsync(((uint8_t*)getDD())+ofs,hostPtr,numBytes);
    }

    // helper function(s)
Backend* Backend::get()
    {
        if (!singleton) singleton = create();
        return singleton;
    }

void Buffer::upload(const void* hostPtr,
        size_t numBytes,
        size_t ofs)
    {
        uploadAsync(hostPtr, numBytes, ofs);
        device->sync();
    }


    
    int Backend::getDeviceCount()
    {
      return get()->numPhysicalDevices;
    }
  
    Backend *Backend::create()
    {
      const char *_fromEnv = getenv("BARNEY_BACKEND");;
      std::string fromEnv = _fromEnv?_fromEnv:"";
#ifndef NDEBUG
      if (fromEnv != "")
        std::cout << "#bn: backend requested from env: '"
                  << fromEnv << "'" << std::endl;
#endif
      if (fromEnv == "") {
        // nothing to do, leave default selection below
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

#if BARNEY_BACKEND_OPTIX
      try {
        return createBackend_optix();
      } catch (std::exception &e) { PING; PRINT(e.what()); }
#endif
      
#if BARNEY_BACKEND_CUDA
      try {
        return createBackend_cuda();
      } catch (std::exception &e) {PING; PRINT(e.what()); }
#endif
      
#if BARNEY_BACKEND_EMBREE
      try {
        return createBackend_embree();
      } catch (std::exception &e) {PING; PRINT(e.what()); }
#endif
      throw std::runtime_error("could not create _any_ backend!?");
      return nullptr;
    }
  



    /*Export a symbol to ask the dynamic loader about in order to locate this library at runtime.*/

    extern "C" int _rtcore_anchor()
    {
        return 0;
    }


        std::string rtcore_symbols_location()
        {
#if defined(_WIN32) && !defined(__CYGWIN__)
            MEMORY_BASIC_INFORMATION mbi;
            VirtualQuery((LPCVOID)&_rtcore_anchor, &mbi, sizeof(mbi));
            char pathBuf[16384];
            if (!GetModuleFileNameA(
                static_cast<HMODULE>(mbi.AllocationBase), pathBuf, sizeof(pathBuf)))
                return std::string();

            std::string path = std::string(pathBuf);
//            path.resize(path.rfind('\\') + 1);
#else
            const char* anchor = "_anari_anchor";
            void* handle = dlsym(RTLD_DEFAULT, anchor);
            if (!handle)
                return std::string();

            Dl_info di;
            int ret = dladdr(handle, &di);
            if (!ret || !di.dli_saddr || !di.dli_fname)
                return std::string();

            std::string path = std::string(di.dli_fname);
            path.resize(path.rfind('/') + 1);
#endif

            return path;
        }


    const void *getSymbol(const std::string &symbolName)
    {
#ifdef _WIN32
        std::string rtCoreLib = rtcore_symbols_location();
        PRINT(rtCoreLib);
        HMODULE libCurrent = GetModuleHandle(rtCoreLib.c_str());
        PRINT(libCurrent);
      void* symbol = (void*)GetProcAddress(libCurrent, symbolName.c_str());
#else
      // void *lib = dlopen(nullptr,RTLD_NOW);
      void *lib = dlopen(nullptr,RTLD_LOCAL|RTLD_NOW);
# ifndef NDEBUG
      if (!lib) std::cout << "#bn: error on dlopen(null): " << dlerror() << std::endl;
# endif
      void* symbol = (void*)dlsym(RTLD_DEFAULT, symbolName.c_str());
      // void* symbol = (void*)dlsym(lib, symbolName.c_str());
# ifndef NDEBUG
      if (!lib) std::cout << "#bn: error on dlsym: " << dlerror() << std::endl;
# endif
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



