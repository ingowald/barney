// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "anari/Device.h"
// anari
#include "anari/backend/LibraryImpl.h"
//#include "generated/anari_library_barney_export.h"

#ifdef ANARI_LIBRARY_BARNEY_STATIC_DEFINE
#  define BARNEY_LIBRARY_INTERFACE
#  define ANARI_LIBRARY_BARNEY_NO_EXPORT
#else
#  ifndef BARNEY_LIBRARY_INTERFACE
#    if defined(_MSC_VER)
#      if defined(anari_library_barney_EXPORTS) || defined(anari_library_barney_mpi_EXPORTS)
/* We are building this library */
#        define BARNEY_LIBRARY_INTERFACE __declspec(dllexport)
#      else
/* We are using this library */
#        define BARNEY_LIBRARY_INTERFACE /* __declspec(dllimport) */
#      endif
#    else
#      if defined(anari_library_barney_EXPORTS) || defined(anari_library_barney_mpi_EXPORTS)
/* We are building this library */
#        define BARNEY_LIBRARY_INTERFACE __attribute__((visibility("default")))
#      else
/* We are using this library */
#        define BARNEY_LIBRARY_INTERFACE __attribute__((visibility("default")))
#      endif
#    endif
#  endif
#endif

// #include "generated/anari_library_barney_queries.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

namespace BARNEY_NS {
  namespace anari {

    // using barney_device::query_extensions;
  
    struct BarneyLibrary : public ::anari::LibraryImpl
    {
      BarneyLibrary(void *lib,
                    ANARIStatusCallback defaultStatusCB,
                    const void *statusCBPtr);

      
      ANARIDevice newDevice(const char *subtype) override;
      const char **getDeviceExtensions(const char *deviceType) override;
    };

    // Definitions ////////////////////////////////////////////////////////////////

    BarneyLibrary::BarneyLibrary(void *lib,
                                 ANARIStatusCallback defaultStatusCB,
                                 const void *statusCBPtr)
      : ::anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
    {}

    ANARIDevice BarneyLibrary::newDevice(const char *subType)
    {
      try {
        return (ANARIDevice) new BarneyDevice(this_library(), subType);
      } catch (std::exception &e) {
        std::cout << "could not create barney device '" << TOSTRING(BARNEY_NS)
                  << ": " << e.what() << std::endl;
        return (ANARIDevice)0;
      }
    }

    const char **BarneyLibrary::getDeviceExtensions(const char *subType)
    {
      // return query_extensions();
      try {
        // BarneyDevice device(this_library(), subType);
        // return device.extensions();
        BarneyDevice *device = (BarneyDevice*)newDevice(subType);
        const char **extensions = device->extensions();
        delete device;
        return extensions;
      } catch (std::exception &e) {
        std::cout << "could not create barney device '" << TOSTRING(BARNEY_NS)
                  << ": " << e.what() << std::endl;
        return 0;
      }
    }

  }
} // namespace BANARI_NS

// have to have this in a auto-generated file because we have to get
// cmake to insert BARNEY_DEVIEC_NAME - can't do that through #defines
// because the entrypoint macro is itself a macro so it wont'
// substitute recursively
#include "Library_entryPoint.h"
