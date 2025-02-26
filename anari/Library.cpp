// Copyright 2023-2024 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Device.h"
// anari
#include "BarneyDeviceQueries.h"
#include "anari/backend/LibraryImpl.h"
#include "anari_library_barney_export.h"

namespace barney_device {

  struct BarneyLibrary : public anari::LibraryImpl
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
    : anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
  {}

  ANARIDevice BarneyLibrary::newDevice(const char *subType)
  {
    return (ANARIDevice) new BarneyDevice(this_library(),subType);
  }

  const char **BarneyLibrary::getDeviceExtensions(const char * /*deviceType*/)
  {
    return query_extensions();
  }

  
} // namespace barney_device

// Define library entrypoint //////////////////////////////////////////////////

extern "C" 
BARNEY_LIBRARY_INTERFACE
ANARI_DEFINE_LIBRARY_ENTRYPOINT(barney, handle, scb, scbPtr)
{
  return (ANARILibrary) new barney_device::BarneyLibrary(handle, scb, scbPtr);
}
