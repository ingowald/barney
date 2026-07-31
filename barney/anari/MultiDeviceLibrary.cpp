// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// anari
#include "anari/backend/LibraryImpl.h"
#include "anari/BaseDevice.h"
#include <map>
#include <string>

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

#define STRINGIFY(x) #x
#define TO_STRING(x) STRINGIFY(x)

namespace BARNEY_LIBRARY_NAME {

  struct BarneyMultiLibrary : public ::anari::LibraryImpl
  {
    BarneyMultiLibrary(void *lib,
                       ANARIStatusCallback defaultStatusCB,
                       const void *statusCBPtr);

    ANARIDevice newDevice(const char *subtype) override;
    const char **getDeviceExtensions(const char *deviceType) override;
  };

  // Definitions ////////////////////////////////////////////////////////////////

  BarneyMultiLibrary::BarneyMultiLibrary(void *lib,
                               ANARIStatusCallback defaultStatusCB,
                               const void *statusCBPtr)
    : ::anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
  {}

#if BARNEY_HAVE_OPTIX
  ANARIDevice createDevice_#ANARI_LIBRARY_NAME#_optix(ANARILibrary,const char *);
#endif
#if BARNEY_HAVE_CPU
  ANARIDevice createDevice_#ANARI_LIBRARY_NAME#_optix(ANARILibrary,const char *);
#endif
  
  ANARIDevice BarneyMultiLibrary::newDevice(const char *subType)
  {
    ANARILibrary lib = this_library();
#if BARNEY_HAVE_OPTIX
    try { createDevice_#ANARI_LIBRARY_NAME#_optix(lib,subType); } catch (...) {};
#endif
#if BARNEY_HAVE_CPU
    try { createDevice_#ANARI_LIBRARY_NAME#_cpu(lib,subType); } catch (...) {};
#endif
// #if BARNEY_HAVE_CUDA
//     try { return createSingleLibraryDevice("cuda",subType); } catch (...) {};
// #endif
// #if BARNEY_HAVE_HIPRT
//     try { return createSingleLibraryDevice("hiprt",subType); } catch (...) {};
// #endif
// #if BARNEY_HAVE_HIP
//     try { return createSingleLibraryDevice("hip",subType); } catch (...) {};
// #endif
// #if BARNEY_HAVE_CPU
//     try { return createSingleLibraryDevice("cpu",subType); } catch (...) {};
// #endif
    // reportMessage(ANARI_SEVERITY_DEBUG, "could not create _any_ barney device !?");
    return (ANARIDevice)0;
  }

  const char **BarneyMultiLibrary::getDeviceExtensions(const char *_deviceType)
  {
    const std::string deviceType
      = _deviceType
      ? _deviceType
      : "default";
    static std::map<std::string,const char **> alreadyFound;
    if (alreadyFound.find(deviceType) != alreadyFound.end())
      return alreadyFound[deviceType];
    ANARIDevice dev = newDevice(deviceType.c_str());
    if (!dev) {
      alreadyFound[deviceType] = nullptr;
      return nullptr;
    }
    BaseDevice *base = (BaseDevice*)dev;
    alreadyFound[deviceType] = base->extensions();

    return alreadyFound[deviceType];
  }

} // namespace BARNEY_LIBRARY_NAME

// have to have this in a auto-generated file because we have to get
// cmake to insert BARNEY_DEVIEC_NAME - can't do that through #defines
// because the entrypoint macro is itself a macro so it wont'
// substitute recursively
// #include "Library_entryPoint_multi.h"

#ifdef BARNEY_MPI
should be off!?
extern "C" BARNEY_LIBRARY_INTERFACE ANARI_DEFINE_LIBRARY_ENTRYPOINT(
                                                                    @BARNEY_DEVICE_NAME@_mpi, handle, scb, scbPtr)
{
  return (ANARILibrary) new BARNEY_NS::anari::BarneyMultiLibrary(handle, scb, scbPtr);
}
#else
extern "C" BARNEY_LIBRARY_INTERFACE
ANARI_DEFINE_LIBRARY_ENTRYPOINT(BARNEY_LIBRARY_NAME, handle, scb, scbPtr)
{
  return (ANARILibrary) new BARNEY_LIBRARY_NAME::BarneyMultiLibrary(handle, scb, scbPtr);
}
#endif
