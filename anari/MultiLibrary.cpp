// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file anari/MultiLibrary.cpp Whereas Library.cpp is specific to a
    device configuration (ie, one for cuda, another for optix, etc),
    the 'MultiLibrary' creates a library that can load differnet
    devices, and by default will try several ones until one works */

#include "Device.h"
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


#include "generated/anari_library_barney_queries.h"

// iw - mind this is BARNEY namespace, NOT banari
namespace BARNEY_NS {
  using barney_device::query_extensions;
  
  struct MultiLibrary : public anari::LibraryImpl
  {
    MultiLibrary(void *lib,
                  ANARIStatusCallback defaultStatusCB,
                  const void *statusCBPtr);

    ANARIDevice newDevice(const char *subtype) override;
    const char **getDeviceExtensions(const char *deviceType) override;
  };

  // Definitions ////////////////////////////////////////////////////////////////

  MultiLibrary::MultiLibrary(void *lib,
                               ANARIStatusCallback defaultStatusCB,
                               const void *statusCBPtr)
    : anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
  {}

  ANARIDevice MultiLibrary::newDevice(const char *subType)
  {
    try {
      return (ANARIDevice) new BarneyDevice(this_library(), subType);
    } catch (std::exception &e) {
      std::cout << "could not create barney device '" << #BARNEY_NS
                << ": " << e.what() << std::endl;
      return (ANARIDevice)0;
    }
  }

  const char **MultiLibrary::getDeviceExtensions(const char * /*deviceType*/)
  {
    return query_extensions();
  }

} // namespace BARNEY_NS

#include "Library_entryPoint.h"
