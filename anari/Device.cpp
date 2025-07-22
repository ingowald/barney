// Copyright 2023-2024 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Device.h"
#if BARNEY_MPI
#include <mpi.h>
#endif

#include "Array.h"
#include "Frame.h"
// std
#include <cstring>

#include "BarneyDeviceQueries.h"

namespace barney_device {

// Data Arrays ////////////////////////////////////////////////////////////////

ANARIArray1D BarneyDevice::newArray1D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems)
{
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

ANARIArray2D BarneyDevice::newArray2D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2)
{
  helium::Array2DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;

  return (ANARIArray2D) new Array2D(deviceState(), md);
}

ANARIArray3D BarneyDevice::newArray3D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t numItems3)
{
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

ANARILight BarneyDevice::newLight(const char *subtype)
{
  return (ANARILight)Light::createInstance(subtype, deviceState());
}

ANARICamera BarneyDevice::newCamera(const char *subtype)
{
  ANARICamera cam = (ANARICamera)Camera::createInstance(subtype, deviceState());
  assert(cam);
  return cam;
}

ANARIGeometry BarneyDevice::newGeometry(const char *subtype)
{
  return (ANARIGeometry)Geometry::createInstance(subtype, deviceState());
}

ANARISpatialField BarneyDevice::newSpatialField(const char *subtype)
{
  return (ANARISpatialField)SpatialField::createInstance(
      subtype, deviceState());
}

ANARISurface BarneyDevice::newSurface()
{
  return (ANARISurface) new Surface(deviceState());
}

ANARIVolume BarneyDevice::newVolume(const char *subtype)
{
  return (ANARIVolume)Volume::createInstance(subtype, deviceState());
}

// Model Meta-Data ////////////////////////////////////////////////////////////

ANARIMaterial BarneyDevice::newMaterial(const char *subtype)
{
  return (ANARIMaterial)Material::createInstance(subtype, deviceState());
}

ANARISampler BarneyDevice::newSampler(const char *subtype)
{
  return (ANARISampler)Sampler::createInstance(subtype, deviceState());
}

// Instancing /////////////////////////////////////////////////////////////////

ANARIGroup BarneyDevice::newGroup()
{
  return (ANARIGroup) new Group(deviceState());
}

ANARIInstance BarneyDevice::newInstance(const char * /*subtype*/)
{
  return (ANARIInstance) new Instance(deviceState());
}

// Top-level Worlds ///////////////////////////////////////////////////////////

ANARIWorld BarneyDevice::newWorld()
{
  return (ANARIWorld) new World(deviceState());
}

// Query functions ////////////////////////////////////////////////////////////

const char **BarneyDevice::getObjectSubtypes(ANARIDataType objectType)
{
  return barney_device::query_object_types(objectType);
}

const void *BarneyDevice::getObjectInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *infoName,
    ANARIDataType infoType)
{
  return barney_device::query_object_info(
      objectType, objectSubtype, infoName, infoType);
}

const void *BarneyDevice::getParameterInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *parameterName,
    ANARIDataType parameterType,
    const char *infoName,
    ANARIDataType infoType)
{
  return barney_device::query_param_info(objectType,
      objectSubtype,
      parameterName,
      parameterType,
      infoName,
      infoType);
}

// Object + Parameter Lifetime Management /////////////////////////////////////

int BarneyDevice::getProperty(ANARIObject object,
    const char *name,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    uint32_t mask)
{
  if (mask == ANARI_WAIT) {
    auto lock = scopeLockObject();
  }

  return helium::BaseDevice::getProperty(object, name, type, mem, size, mask);
}

// Frame Manipulation /////////////////////////////////////////////////////////

ANARIFrame BarneyDevice::newFrame()
{
  return (ANARIFrame) new Frame(deviceState());
}

// Frame Rendering ////////////////////////////////////////////////////////////

ANARIRenderer BarneyDevice::newRenderer(const char *)
{
  return (ANARIRenderer) new Renderer(deviceState());
}

// Helper/other functions and data members ////////////////////////////////////

std::vector<std::string> splitString(std::string s, const char delim)
{
  std::vector<std::string> res;
  while (true) {
    size_t pos = s.find(delim);
    if (pos == s.npos)
      break;
    res.push_back(s.substr(0, pos));
    s = s.substr(pos + 1);
  }
  res.push_back(s);
  return res;
}

static void default_statusFunc(const void * /*userData*/,
    ANARIDevice /*device*/,
    ANARIObject source,
    ANARIDataType /*sourceType*/,
    ANARIStatusSeverity severity,
    ANARIStatusCode /*code*/,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL][%p] %s\n", source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
#ifndef NDEBUG
  } else if (severity == ANARI_SEVERITY_WARNING) {
    fprintf(stderr, "[WARN ][%p] %s\n", source, message);
  } else if (severity == ANARI_SEVERITY_PERFORMANCE_WARNING) {
    fprintf(stderr, "[PERF ][%p] %s\n", source, message);
#endif
  }
  // Ignore INFO/DEBUG messages
}

BarneyDevice::BarneyDevice(ANARILibrary l, const std::string &subType)
    : helium::BaseDevice(l), deviceType(subType)
{
  std::vector<std::string> subTypeFlags = splitString(subType, ',');
  for (auto flag : subTypeFlags) {
    if (flag == "cpu") {
      m_cudaDevice = -1;
      continue;
    }
    if (flag == "default") {
#if BARNEY_MPI
      comm = MPI_COMM_WORLD;
#endif
      continue;
    }
#if BARNEY_MPI
    if (flag == "local") {
      comm = 0;
      continue;
    }
    if (flag == "mpi") {
      comm = MPI_COMM_WORLD;
      continue;
    }
#endif
    std::cout << "un-recognized feature '" << flag << "' on device subtype"
              << std::endl;
    // reportMessage(ANARI_SEVERITY_WARNING,
    //               "un-recognized feature '%s' on device subtype",
    //               flag.c_str());
  }

  m_state = std::make_unique<BarneyGlobalState>(this_device());
}

BarneyDevice::BarneyDevice() : helium::BaseDevice(default_statusFunc, nullptr)
{
  m_state = std::make_unique<BarneyGlobalState>(this_device());
}

/*! helper entry-point for _directly_ creating a banari device
  without having to go through the dynamic-library
  'anariLoadLibrary' mechanism. This is used in pynari, to allow
  static linking of anari sdk */
extern "C" ANARIDevice createAnariDeviceBarney()
{
  ANARIDevice dev = 0;
  try {
    // int numGPUs = 0;
    // bnCountAvailableDevice(&numGPUs);
    // if (numGPUs == 0)
    //   throw std::runtime_error("#barney/anari: cannot create device - no
    //   GPUs?");
    dev = (ANARIDevice) new BarneyDevice();
    return dev;
  } catch (std::exception &err) {
    std::cerr << "#banari: exception creating anari 'barney' GPU device: "
              << err.what() << std::endl;
    //    throw;
    return 0;
  }
}

BarneyDevice::~BarneyDevice()
{
  auto &state = *deviceState();
  state.commitBuffer.clear();
  reportMessage(ANARI_SEVERITY_DEBUG, "destroying barney device (%p)", this);
}

void BarneyDevice::initDevice()
{
  if (m_initialized)
    return;

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing barney device (%p)", this);

  auto state = deviceState();

  try {
    int rank = 0, size = 1;
#if BARNEY_MPI
    if (comm != 0) {
      int mpiInitialized = 0;
      MPI_Initialized(&mpiInitialized);
      if (!mpiInitialized)
        MPI_Init(nullptr, nullptr);
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &size);
    } else
      reportMessage(ANARI_SEVERITY_DEBUG,
          "app passed null communicator, falling back to local rendering");
#endif
    std::vector<int> gpuIDs;
    int *_gpuIDs = nullptr;
    int _gpuCount = -1;
    if (state->tether->devices[0]->m_cudaDevice >= 0) {
      for (auto dev : state->tether->devices) {
        assert(dev->m_cudaDevice >= 0);
        gpuIDs.push_back(dev->m_cudaDevice);
      }
      _gpuIDs = gpuIDs.data();
      _gpuCount = (int)gpuIDs.size();
    } else {
      // leave empty, init with barney gpu list with {nullptr,-1}
    }

    std::vector<int> dgIDs;
    if (state->tether->devices[0]->m_dataGroupID >= 0) {
      for (auto dev : state->tether->devices) {
        assert(dev->m_dataGroupID >= 0);
        dgIDs.push_back(dev->m_dataGroupID);
      }
    } else {
      dgIDs = {rank};
    }
    int *_dgIDs = dgIDs.data();
    int _dgCount = (int)dgIDs.size();

    reportMessage(ANARI_SEVERITY_DEBUG, "using cuda device #%i", m_cudaDevice);

#if BARNEY_MPI
    if (comm)
      state->tether->context =
          bnMPIContextCreate(comm, _dgIDs, _dgCount, _gpuIDs, _gpuCount);
    else
#endif
      state->tether->context =
          bnContextCreate(_dgIDs, _dgCount, _gpuIDs, _gpuCount);

    std::stringstream ss;
    ss << "#banari rank " << rank << " (of " << size
       << ") creating context GPUs=(";
    for (auto gpu : gpuIDs)
      ss << " " << gpu;
    ss << " ) and data groups=(";
    for (auto dg : dgIDs)
      ss << " " << dg;
    ss << " )";

    std::cout << ss.str() << std::endl;
    reportMessage(ANARI_SEVERITY_DEBUG, ss.str().c_str());
    m_initialized = true;
  } catch (const std::exception &err) {
    std::cerr
        << "#banari: ran into some kind of exception in barney device init"
        << err.what() << std::endl;
  }
}

void BarneyDevice::deviceCommitParameters()
{
  helium::BaseDevice::deviceCommitParameters();

  auto state = deviceState(false);
  if (state->hasBeenCommitted) {
    reportMessage(ANARI_SEVERITY_DEBUG, "device committed more than once!");
  } else {
    state->hasBeenCommitted = true;
  }
  m_cudaDevice = getParam<int>("cudaDevice", m_cudaDevice);
  m_dataGroupID = getParam<int>("dataGroupID", m_dataGroupID);
#if BARNEY_MPI
  static_assert(sizeof(void*) == sizeof(MPI_Comm),
                "we assume an MPI_Comm to be a pointer type, seems for this MPI implementation that's not the case. Pls let the developers know what MPI flavor and version you're using so this can be fixed.");
  uint64_t passedComm = getParam<uint64_t>("pointer_to_mpi_communicator", 0ull);//MPI_COMM_WORLD);
  if (passedComm) {
    printf("#banari.mpi: got passed a pointer to a MPI communicator, going to use this.\n");
    comm = *(MPI_Comm *)passedComm;
  } else
    comm = MPI_COMM_WORLD;
  // (void *&)comm = getParam<void *>("communicator", (void*)0);//MPI_COMM_WORLD);
  if (comm) {
    int rank, size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
    printf("#banari.mpi: running data parallel on rank %i size %i\n",
           rank,size);
  }
#endif
  if (m_cudaDevice != -2)
    std::cout << "#banari: found 'cudaDevice' = " << m_cudaDevice << std::endl;

  if (!state->tether) {
    tetherIndex = getParam<int>("tetherIndex", tetherIndex);
    tetherCount = getParam<int>("tetherCount", 1);
    state->slot = tetherIndex;
    auto tetherDev = getParam<anari::Device>("tetherDevice", (anari::Device)0);
    tetherDevice = (BarneyDevice *)tetherDev;

    assert(tetherCount > 0);
    assert(tetherIndex >= 0);
    assert(tetherIndex < tetherCount);

#ifndef NDEBUG
    std::cout << "#banari: FIRST-TIME device initialization slot "
              << tetherIndex << "/" << tetherCount << " in tethered dev "
              << tetherDevice << std::endl;
#endif
    if (tetherDevice) {
#ifndef NDEBUG
      std::cout << "#banari -> tethering to primary" << std::endl;
#endif
      state->tether = tetherDevice->deviceState()->tether;
      assert(state->tether);
      assert(tetherCount == state->tether->devices.size());
      assert(state->tether->devices[tetherIndex] == nullptr);
      state->tether->devices[tetherIndex] = this;
    } else {
      assert(tetherIndex == 0 && "first device has to be first to be created");
      state->tether = std::make_shared<Tether>();
      state->tether->devices.resize(tetherCount);
      state->tether->devices[0] = this;
    }
    if (state->tether->allDevicesPresent()) {
      assert(state->tether->context == 0
          && "only last device to tether should create the barney context");
      initDevice();
    }
  }
}

int BarneyDevice::deviceGetProperty(const char *name,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    uint32_t mask)
{
  std::string_view prop = name;
  if (prop == "extension" && type == ANARI_STRING_LIST) {
    helium::writeToVoidP(mem, query_extensions());
    return 1;
  } else if (prop == "barney" && type == ANARI_BOOL) {
    helium::writeToVoidP(mem, true);
    return 1;
  }
  return 0;
}

BarneyGlobalState *BarneyDevice::deviceState(bool commitOnDemand)
{
  BarneyGlobalState *state =
      (BarneyGlobalState *)helium::BaseDevice::m_state.get();
  if (commitOnDemand && !state->hasBeenCommitted)
    deviceCommitParameters();
  return state;
}

} // namespace barney_device
