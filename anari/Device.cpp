// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Device.h"
#if TALLY_MPI
#include <mpi.h>
#endif

#include "Array.h"
#include "Frame.h"
// std
#include <cstring>

#include "TallyDeviceQueries.h"

namespace tally_device {

// Data Arrays ////////////////////////////////////////////////////////////////

void *TallyDevice::mapArray(ANARIArray a)
{
  deviceState()->waitOnCurrentFrame();
  return helium::BaseDevice::mapArray(a);
}

ANARIArray1D TallyDevice::newArray1D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems)
{
  initDevice();

  helium::Array1DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems = numItems;

  if (anari::isObject(type))
    return (ANARIArray1D) new ObjectArray(deviceState(), md);
  else
    return (ANARIArray1D) new Array1D(deviceState(), md);
}

ANARIArray2D TallyDevice::newArray2D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2)
{
  initDevice();

  helium::Array2DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;

  return (ANARIArray2D) new Array2D(deviceState(), md);
}

ANARIArray3D TallyDevice::newArray3D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t numItems3)
{
  initDevice();

  helium::Array3DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;
  md.numItems3 = numItems3;

  return (ANARIArray3D) new Array3D(deviceState(), md);
}

// Renderable Objects /////////////////////////////////////////////////////////

ANARILight TallyDevice::newLight(const char *subtype)
{
  initDevice();
  return (ANARILight)Light::createInstance(subtype, deviceState());
}

ANARICamera TallyDevice::newCamera(const char *subtype)
{
  initDevice();
  return (ANARICamera)Camera::createInstance(subtype, deviceState());
}

ANARIGeometry TallyDevice::newGeometry(const char *subtype)
{
  initDevice();
  return (ANARIGeometry)Geometry::createInstance(subtype, deviceState());
}

ANARISpatialField TallyDevice::newSpatialField(const char *subtype)
{
  initDevice();
  return (ANARISpatialField)SpatialField::createInstance(
      subtype, deviceState());
}

ANARISurface TallyDevice::newSurface()
{
  initDevice();
  return (ANARISurface) new Surface(deviceState());
}

ANARIVolume TallyDevice::newVolume(const char *subtype)
{
  initDevice();
  return (ANARIVolume)Volume::createInstance(subtype, deviceState());
}

// Model Meta-Data ////////////////////////////////////////////////////////////

ANARIMaterial TallyDevice::newMaterial(const char *subtype)
{
  initDevice();
  return (ANARIMaterial)Material::createInstance(subtype, deviceState());
}

ANARISampler TallyDevice::newSampler(const char *subtype)
{
  initDevice();
  return (ANARISampler)Sampler::createInstance(subtype, deviceState());
}

// Instancing /////////////////////////////////////////////////////////////////

ANARIGroup TallyDevice::newGroup()
{
  initDevice();
  return (ANARIGroup) new Group(deviceState());
}

ANARIInstance TallyDevice::newInstance(const char * /*subtype*/)
{
  initDevice();
  return (ANARIInstance) new Instance(deviceState());
}

// Top-level Worlds ///////////////////////////////////////////////////////////

ANARIWorld TallyDevice::newWorld()
{
  initDevice();
  return (ANARIWorld) new World(deviceState());
}

// Query functions ////////////////////////////////////////////////////////////

const char **TallyDevice::getObjectSubtypes(ANARIDataType objectType)
{
  return tally_device::query_object_types(objectType);
}

const void *TallyDevice::getObjectInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *infoName,
    ANARIDataType infoType)
{
  return tally_device::query_object_info(
      objectType, objectSubtype, infoName, infoType);
}

const void *TallyDevice::getParameterInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *parameterName,
    ANARIDataType parameterType,
    const char *infoName,
    ANARIDataType infoType)
{
  return tally_device::query_param_info(objectType,
      objectSubtype,
      parameterName,
      parameterType,
      infoName,
      infoType);
}

// Object + Parameter Lifetime Management /////////////////////////////////////

int TallyDevice::getProperty(ANARIObject object,
    const char *name,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    uint32_t mask)
{
  if (mask == ANARI_WAIT) {
    auto lock = scopeLockObject();
    deviceState()->waitOnCurrentFrame();
  }

  return helium::BaseDevice::getProperty(object, name, type, mem, size, mask);
}

// Frame Manipulation /////////////////////////////////////////////////////////

ANARIFrame TallyDevice::newFrame()
{
  initDevice();
  return (ANARIFrame) new Frame(deviceState());
}

// Frame Rendering ////////////////////////////////////////////////////////////

ANARIRenderer TallyDevice::newRenderer(const char *)
{
  initDevice();
  return (ANARIRenderer) new Renderer(deviceState());
}

// Helper/other functions and data members ////////////////////////////////////

TallyDevice::TallyDevice(ANARILibrary l) : helium::BaseDevice(l)
{
  m_state = std::make_unique<TallyGlobalState>(this_device());
  deviceCommitParameters();
}

TallyDevice::~TallyDevice()
{
  auto &state = *deviceState();
  state.commitBufferClear();
  reportMessage(ANARI_SEVERITY_DEBUG, "destroying tally device (%p)", this);
}

void TallyDevice::initDevice()
{
  if (m_initialized)
    return;

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing tally device (%p)", this);

  auto &state = *deviceState();

#if TALLY_MPI
  int mpiInitialized = 0;
  MPI_Initialized(&mpiInitialized);
  if (!mpiInitialized)
    MPI_Init(nullptr, nullptr);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  state.context = bnMPIContextCreate(MPI_COMM_WORLD, &rank, 1, nullptr, 0);

  auto &info = state.bnInfo;
  bnMPIQueryHardware(&info, MPI_COMM_WORLD);
  reportMessage(ANARI_SEVERITY_DEBUG, "BNHardwareInfo:");
  reportMessage(ANARI_SEVERITY_DEBUG, "    numRanks: %i", info.numRanks);
  reportMessage(ANARI_SEVERITY_DEBUG, "    numHosts: %i", info.numHosts);
  reportMessage(
      ANARI_SEVERITY_DEBUG, "    numGPUsThisRank: %i", info.numGPUsThisRank);
  reportMessage(
      ANARI_SEVERITY_DEBUG, "    numGPUsThisHost: %i", info.numGPUsThisHost);
  reportMessage(
      ANARI_SEVERITY_DEBUG, "    numRanksThisHost: %i", info.numRanksThisHost);
  reportMessage(ANARI_SEVERITY_DEBUG, "    localRank: %i", info.localRank);
#else
  state.context = bnContextCreate();
  std::memset(&state.bnInfo, 0, sizeof(state.bnInfo));
#endif
  reportMessage(
      ANARI_SEVERITY_DEBUG, "created tally context (%p)", state.context);

  m_initialized = true;
}

void TallyDevice::deviceCommitParameters()
{
  auto &state = *deviceState();

  bool allowInvalidSurfaceMaterials = state.allowInvalidSurfaceMaterials;

  state.allowInvalidSurfaceMaterials =
      getParam<bool>("allowInvalidMaterials", true);
  state.invalidMaterialColor = getParam<math::float4>(
      "invalidMaterialColor", math::float4(1.f, 0.f, 1.f, 1.f));

  if (allowInvalidSurfaceMaterials != state.allowInvalidSurfaceMaterials)
    state.objectUpdates.lastSceneChange = helium::newTimeStamp();

  helium::BaseDevice::deviceCommitParameters();
}

int TallyDevice::deviceGetProperty(
    const char *name, ANARIDataType type, void *mem, uint64_t size)
{
  std::string_view prop = name;
  if (prop == "extension" && type == ANARI_STRING_LIST) {
    helium::writeToVoidP(mem, query_extensions());
    return 1;
  } else if (prop == "tally" && type == ANARI_BOOL) {
    helium::writeToVoidP(mem, true);
    return 1;
  }
  return 0;
}

TallyGlobalState *TallyDevice::deviceState() const
{
  return (TallyGlobalState *)helium::BaseDevice::m_state.get();
}

} // namespace tally_device
