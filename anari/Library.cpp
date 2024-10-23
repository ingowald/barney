// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Device.h"
// anari
#include "TallyDeviceQueries.h"
#include "anari/backend/LibraryImpl.h"
#include "anari_library_barney_export.h"

namespace tally_device {

struct TallyLibrary : public anari::LibraryImpl
{
  TallyLibrary(
      void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr);

  ANARIDevice newDevice(const char *subtype) override;
  const char **getDeviceExtensions(const char *deviceType) override;
};

// Definitions ////////////////////////////////////////////////////////////////

TallyLibrary::TallyLibrary(
    void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr)
    : anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
{}

ANARIDevice TallyLibrary::newDevice(const char * /*subtype*/)
{
  return (ANARIDevice) new TallyDevice(this_library());
}

const char **TallyLibrary::getDeviceExtensions(const char * /*deviceType*/)
{
  return query_extensions();
}

} // namespace tally_device

// Define library entrypoint //////////////////////////////////////////////////

extern "C" BARNEY_LIBRARY_INTERFACE ANARI_DEFINE_LIBRARY_ENTRYPOINT(
    barney, handle, scb, scbPtr)
{
  return (ANARILibrary) new tally_device::TallyLibrary(handle, scb, scbPtr);
}
 
