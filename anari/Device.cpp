// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Device.h"

#include "Array.h"
#include "Frame.h"
// std
#include <cstring>

#include "BarneyDeviceQueries.h"

namespace barney_device {

// Data Arrays ////////////////////////////////////////////////////////////////

void *BarneyDevice::mapArray(ANARIArray a)
{
  deviceState()->waitOnCurrentFrame();
  return helium::BaseDevice::mapArray(a);
}

ANARIArray1D BarneyDevice::newArray1D(const void *appMemory,
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

ANARIArray2D BarneyDevice::newArray2D(const void *appMemory,
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

ANARIArray3D BarneyDevice::newArray3D(const void *appMemory,
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

ANARILight BarneyDevice::newLight(const char *)
{
  initDevice();
  return (ANARILight) new UnknownObject(ANARI_LIGHT, deviceState());
}

ANARICamera BarneyDevice::newCamera(const char *subtype)
{
  initDevice();
  return (ANARICamera)Camera::createInstance(subtype, deviceState());
}

ANARIGeometry BarneyDevice::newGeometry(const char *subtype)
{
  initDevice();
  return (ANARIGeometry)Geometry::createInstance(subtype, deviceState());
}

ANARISpatialField BarneyDevice::newSpatialField(const char *subtype)
{
  initDevice();
  return (ANARISpatialField)SpatialField::createInstance(
      subtype, deviceState());
}

ANARISurface BarneyDevice::newSurface()
{
  initDevice();
  return (ANARISurface) new Surface(deviceState());
}

ANARIVolume BarneyDevice::newVolume(const char *subtype)
{
  initDevice();
  return (ANARIVolume)Volume::createInstance(subtype, deviceState());
}

// Model Meta-Data ////////////////////////////////////////////////////////////

ANARIMaterial BarneyDevice::newMaterial(const char *subtype)
{
  initDevice();
  return (ANARIMaterial)Material::createInstance(subtype, deviceState());
}

ANARISampler BarneyDevice::newSampler(const char *)
{
  initDevice();
  return (ANARISampler) new UnknownObject(ANARI_SAMPLER, deviceState());
}

// Instancing /////////////////////////////////////////////////////////////////

ANARIGroup BarneyDevice::newGroup()
{
  initDevice();
  return (ANARIGroup) new Group(deviceState());
}

ANARIInstance BarneyDevice::newInstance(const char * /*subtype*/)
{
  initDevice();
  return (ANARIInstance) new Instance(deviceState());
}

// Top-level Worlds ///////////////////////////////////////////////////////////

ANARIWorld BarneyDevice::newWorld()
{
  initDevice();
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
    deviceState()->waitOnCurrentFrame();
  }

  return helium::BaseDevice::getProperty(object, name, type, mem, size, mask);
}

// Frame Manipulation /////////////////////////////////////////////////////////

ANARIFrame BarneyDevice::newFrame()
{
  initDevice();
  return (ANARIFrame) new Frame(deviceState());
}

// Frame Rendering ////////////////////////////////////////////////////////////

ANARIRenderer BarneyDevice::newRenderer(const char *)
{
  initDevice();
  return (ANARIRenderer) new Renderer(deviceState());
}

// Helper/other functions and data members ////////////////////////////////////

BarneyDevice::BarneyDevice(ANARILibrary l) : helium::BaseDevice(l)
{
  m_state = std::make_unique<BarneyGlobalState>(this_device());
  deviceCommitParameters();
}

BarneyDevice::~BarneyDevice()
{
  auto &state = *deviceState();
  state.commitBufferClear();
  reportMessage(ANARI_SEVERITY_DEBUG, "destroying barney device (%p)", this);
}

void BarneyDevice::initDevice()
{
  if (m_initialized)
    return;

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing barney device (%p)", this);

  auto &state = *deviceState();

  state.context = bnContextCreate();
  reportMessage(
      ANARI_SEVERITY_DEBUG, "created barney context (%p)", state.context);

  m_initialized = true;
}

void BarneyDevice::deviceCommitParameters()
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

int BarneyDevice::deviceGetProperty(
    const char *name, ANARIDataType type, void *mem, uint64_t size)
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

BarneyGlobalState *BarneyDevice::deviceState() const
{
  return (BarneyGlobalState *)helium::BaseDevice::m_state.get();
}

} // namespace barney_device
