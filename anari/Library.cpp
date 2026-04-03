// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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

namespace barney_device {

struct BarneyLibrary : public anari::LibraryImpl
{
  BarneyLibrary(
      void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr);

  ANARIDevice newDevice(const char *subtype) override;
  const char **getDeviceExtensions(const char *deviceType) override;
};

// Definitions ////////////////////////////////////////////////////////////////

BarneyLibrary::BarneyLibrary(
    void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr)
    : anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
{}

ANARIDevice BarneyLibrary::newDevice(const char *subType)
{
  return (ANARIDevice) new BarneyDevice(this_library(), subType);
}

const char **BarneyLibrary::getDeviceExtensions(const char * /*deviceType*/)
{
  return query_extensions();
}

} // namespace barney_device

#include "Library_entryPoint.h"
