# ======================================================================== #
# Copyright 2023-2024 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

message("CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}")

if(MSVC) # That's also clang-cl
  # /bigobj is needed for bigger binding projects due to the limit to 64k
  # addressable sections
  set_property(
    TARGET barney_config
    APPEND
    PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif()



add_library(barney_config_ptx INTERFACE)
target_link_libraries(barney_config_ptx
  INTERFACE
  barney_config
  )
target_compile_definitions(barney_config_ptx
  INTERFACE
  -DBARNEY_COMPILE_OPTIX_PROGRAMS=1
  )


# ------------------------------------------------------------------
# barney itself
# ------------------------------------------------------------------
add_library(barney SHARED
  api/barney.cpp
  )
target_link_libraries(barney PUBLIC
  $<BUILD_INTERFACE:barney_config>
  )
set_target_properties(barney
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION   ON
  POSITION_INDEPENDENT_CODE    ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CUDA_RESOLVE_DEVICE_SYMBOLS  ON
  CXX_VISIBILITY_PRESET  default
  CUDA_VISIBILITY_PRESET default
  )
if (APPLE)
  set_target_properties(barney PROPERTIES INSTALL_RPATH "$loader_path")
else()
  set_target_properties(barney PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
  
add_library(barney_static STATIC
  api/barney.cpp
  )
target_link_libraries(barney_static PUBLIC
  $<BUILD_INTERFACE:barney_config>
#  ${BARNEY_DEPENDS}
  )
set_target_properties(barney_static
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION   ON
  POSITION_INDEPENDENT_CODE    ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CUDA_RESOLVE_DEVICE_SYMBOLS  ON
  CXX_VISIBILITY_PRESET  default
  CUDA_VISIBILITY_PRESET default
  )
if (APPLE)
  set_target_properties(barney PROPERTIES INSTALL_RPATH "$loader_path")
else()
  set_target_properties(barney PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()




set(HOST_SOURCES
  umesh/common/UMeshField.h
  umesh/common/UMeshField.cu
  umesh/mc/UMeshCUBQLSampler.h
  umesh/mc/UMeshCUBQLSampler.cu

  umesh/os/AWT.cu
  umesh/os/AWT.h
  
  # *structured* volumes
  volume/StructuredData.h
  volume/StructuredData.cu
  
  # actual geometry types
  geometry/Geometry.h
  geometry/Geometry.cpp
  geometry/Triangles.h
  geometry/Triangles.cpp
  geometry/Cylinders.h
  geometry/Cylinders.cpp
  geometry/Capsules.h
  geometry/Capsules.cpp
  geometry/Spheres.h
  geometry/Spheres.cpp
  #  geometry/Cones.h
  #  geometry/Cones.cpp
  # volume data 
  volume/Volume.h
  volume/Volume.cpp
  volume/ScalarField.h
  volume/ScalarField.cpp
  volume/MCAccelerator.h
  volume/TransferFunction.h
  volume/TransferFunction.cpp
  volume/MCGrid.h
  volume/MCGrid.cu
  volume/MCAccelerator.h
  
  # samplers
  render/Sampler.h
  render/Sampler.cpp
  
  common/barney-common.h
  
  # actual device kernels for CUDA ray gen and shading
  kernels/generateRays.cu
  kernels/shadeRays.cu
  kernels/traceRays.cu
  # all the barney "plumbing" stuff on the host
  DeviceContext.h
  DeviceContext.cpp
  Context.h
  Context.cpp
  DeviceGroup.h
  DeviceGroup.cpp
  LocalContext.h
  LocalContext.cpp
  Object.h
  Object.cpp
  Camera.h
  Camera.cu
  # all frame buffer related stuff (DistFB in barney_mpi)
  fb/FrameBuffer.h
  fb/FrameBuffer.cu
  fb/LocalFB.h
  fb/LocalFB.cpp
  fb/TiledFB.h
  fb/TiledFB.cu
  # model/group/data group handling
  GlobalModel.h
  GlobalModel.cpp
  render/Renderer.h
  render/Renderer.cpp
  Group.h
  Group.cpp
  render/GeometryAttributes.h
  render/GeometryAttributes.cpp

  render/RayQueue.h
  render/RayQueue.cpp
  render/World.h
  render/World.cpp
  render/MaterialRegistry.h
  render/MaterialRegistry.cpp
  render/SamplerRegistry.h
  render/SamplerRegistry.cpp
  
  #  material/Globals.h
  #  material/Globals.cu
  ModelSlot.h
  ModelSlot.cpp
  # general csommon/data-related stuff
  common/Texture.h
  common/Texture.cpp
  common/Data.h
  common/Data.cpp
  # lights
  light/Light.h
  light/Light.cpp
  light/DirLight.h
  light/DirLight.cu
  light/PointLight.h
  light/PointLight.cu
  light/QuadLight.h
  light/QuadLight.cu
  light/EnvMap.h
  light/EnvMap.cu

  # materials
  material/Material.h
  material/Material.cpp
  material/AnariPBR.h
  material/AnariPBR.cpp
  material/AnariMatte.h
  material/AnariMatte.cpp
  )

set(DEVICE_PROGRAM_SOURCES
  geometry/Spheres.dev.cu
  geometry/Triangles.dev.cu
  geometry/Cylinders.dev.cu
  geometry/Capsules.dev.cu
  volume/StructuredData.dev.cu
  umesh/mc/UMeshMC.dev.cu
  umesh/os/AWT.dev.cu
  amr/BlockStructuredMC.dev.cu
  )


if (BARNEY_BACKEND_OPTIX)
  add_library(barney_optix ${HOST_SOURCES})
  foreach(src  ${DEVICE_PROGRAM_SOURCES})
    get_filename_component(basename "${src}" NAME_WE)
    embed_ptx(
      OUTPUT_TARGET      barney-${basename}-ptx
      PTX_LINK_LIBRARIES barney_config_ptx barney_rtc_optix
      SOURCES            ${src}
      )
    target_link_libraries(barney_optix PRIVATE barney-${basename}-ptx)
  endforeach()
  target_link_libraries(barney_optix PRIVATE barney_config barney_rtc_optix)
  target_link_libraries(barney PRIVATE barney_optix)
endif()
if (BARNEY_BACKEND_EMBREE)
  add_library(barney_embree ${HOST_SOURCES} ${DEVICE_PROGRAM_SOURCES})
  if (NOT BARNEY_HAVE_CUDA)
    foreach(src ${HOST_SOURCES})
      set_source_files_properties(${src} PROPERTIES LANGUAGE CXX)
    endforeach()
  endif()
  target_link_libraries(barney_embree PRIVATE barney_config barney_rtc_embree)
  target_link_libraries(barney PRIVATE barney_embree)
endif()
    













if (OFF)

# -------------------------------------------------------

if (BARNEY_BACKEND_OPTIX)
  message("embedding programs for optix backend")
  cmake_policy(SET CMP0076 NEW)
  add_library(barney_config_ptx INTERFACE)
  target_link_libraries(barney_config_ptx
    INTERFACE
    barney_config
    )
  target_compile_definitions(barney_config_ptx
    INTERFACE
    -DBARNEY_COMPILE_OPTIX_PROGRAMS=1
    )
  
  # ray gen program that launches kernel to trace rays
  embed_ptx(
    OUTPUT_TARGET      barney-traceRays-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/traceRays.cu
    )
  # ------------------------------------------------------------------
  # surface geometry:
  # ------------------------------------------------------------------
  # ray-sphere geom intersection
  embed_ptx(
    OUTPUT_TARGET      barney-spheres-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            ${CMAKE_CURRENT_SOURCE_DIR}/geometry/Spheres.dev.cu
    )
  # ray triangles geom intersection
  embed_ptx(
    OUTPUT_TARGET      barney-triangles-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            ${CMAKE_CURRENT_SOURCE_DIR}/geometry/Triangles.dev.cu
    )
  # ray cylinders geom intersection
  embed_ptx(
    OUTPUT_TARGET      barney-cylinders-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            ${CMAKE_CURRENT_SOURCE_DIR}/geometry/Cylinders.dev.cu
    )
  # ray capsules geom intersection
  embed_ptx(
    OUTPUT_TARGET      barney-capsules-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            ${CMAKE_CURRENT_SOURCE_DIR}/geometry/Capsules.dev.cu
    )
  # ray cones geom intersection
  #embed_ptx(
  #  OUTPUT_TARGET      barney-cones-ptx
  #  PTX_LINK_LIBRARIES barney_config
  #  SOURCES            geometry/Cones.dev.cu
  #  )

  # ------------------------------------------------------------------
  # volume variants:
  # ------------------------------------------------------------------
  embed_ptx(
    OUTPUT_TARGET      barney-StructuredData-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            ${CMAKE_CURRENT_SOURCE_DIR}/volume/StructuredData.dev.cu
#    EMBEDDED_SYMBOL_NAMES MCAccel_Structured_ptx
    )
#  embed_ptx(
#    OUTPUT_TARGET      barney-UMeshRTXObjectSpace-ptx
#    PTX_LINK_LIBRARIES barney_config_ptx
#    SOURCES            umesh/os/RTXObjectSpace.dev.cu
#    )
  embed_ptx(
    OUTPUT_TARGET      barney-UMeshMC-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            ${CMAKE_CURRENT_SOURCE_DIR}/umesh/mc/UMeshMC.dev.cu
 #   EMBEDDED_SYMBOL_NAMES MCAccel_UMesh_ptx
    )
  embed_ptx(
    OUTPUT_TARGET      barney-AWT-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            ${CMAKE_CURRENT_SOURCE_DIR}/umesh/os/AWT.dev.cu
    )

  # ------------------------------------------------------------------
  # amr variants:
  # ------------------------------------------------------------------
  embed_ptx(
    OUTPUT_TARGET      barney-BlockStructuredMC-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            ${CMAKE_CURRENT_SOURCE_DIR}/amr/BlockStructuredMC.dev.cu
    )
endif()



set(BARNEY_SOURCES
  geometry/Triangles.dev.cu
  geometry/Capsules.dev.cu
  geometry/Spheres.dev.cu
  
  # unstructured meshes
  umesh/common/UMeshField.h
  umesh/common/UMeshField.cu
  umesh/mc/UMeshCUBQLSampler.h
  umesh/mc/UMeshCUBQLSampler.cu
  umesh/mc/UMeshMC.dev.cu

  umesh/os/AWT.dev.cu
  umesh/os/AWT.cu
  umesh/os/AWT.h
  
  # *structured* volumes
  volume/StructuredData.h
  volume/StructuredData.cu
  volume/StructuredData.dev.cu
  
  # actual geometry types
  geometry/Geometry.h
  geometry/Geometry.cpp
  geometry/Triangles.h
  geometry/Triangles.cpp
  geometry/Cylinders.h
  geometry/Cylinders.cpp
  geometry/Capsules.h
  geometry/Capsules.cpp
  geometry/Spheres.h
  geometry/Spheres.cpp
  #  geometry/Cones.h
  #  geometry/Cones.cpp
  # volume data 
  volume/Volume.h
  volume/Volume.cpp
  volume/ScalarField.h
  volume/ScalarField.cpp
  volume/MCAccelerator.h
  volume/TransferFunction.h
  volume/TransferFunction.cpp
  volume/MCGrid.h
  volume/MCGrid.cu
  volume/MCAccelerator.h
  
  # samplers
  render/Sampler.h
  render/Sampler.cpp
  
  common/barney-common.h
  
  # actual device kernels for CUDA ray gen and shading
  kernels/generateRays.cu
  kernels/shadeRays.cu
  kernels/traceRays.cu
  # all the barney "plumbing" stuff on the host
  DeviceContext.h
  DeviceContext.cpp
  Context.h
  Context.cpp
  DeviceGroup.h
  DeviceGroup.cpp
  LocalContext.h
  LocalContext.cpp
  Object.h
  Object.cpp
  Camera.h
  Camera.cu
  # all frame buffer related stuff (DistFB in barney_mpi)
  fb/FrameBuffer.h
  fb/FrameBuffer.cu
  fb/LocalFB.h
  fb/LocalFB.cpp
  fb/TiledFB.h
  fb/TiledFB.cu
  # model/group/data group handling
  GlobalModel.h
  GlobalModel.cpp
  render/Renderer.h
  render/Renderer.cpp
  Group.h
  Group.cpp
  render/GeometryAttributes.h
  render/GeometryAttributes.cpp

  render/RayQueue.h
  render/RayQueue.cpp
  render/World.h
  render/World.cpp
  render/MaterialRegistry.h
  render/MaterialRegistry.cpp
  render/SamplerRegistry.h
  render/SamplerRegistry.cpp
  
  #  material/Globals.h
  #  material/Globals.cu
  ModelSlot.h
  ModelSlot.cpp
  # general common/data-related stuff
  common/Texture.h
  common/Texture.cpp
  common/Data.h
  common/Data.cpp
  # lights
  light/Light.h
  light/Light.cpp
  light/DirLight.h
  light/DirLight.cu
  light/PointLight.h
  light/PointLight.cu
  light/QuadLight.h
  light/QuadLight.cu
  light/EnvMap.h
  light/EnvMap.cu

  # materials
  material/Material.h
  material/Material.cpp
  material/AnariPBR.h
  material/AnariPBR.cpp
  material/AnariMatte.h
  material/AnariMatte.cpp

  # implements the c99 barney api calls
  barney.cpp
  )


# ==================================================================
if (BARNEY_HAVE_CUDA)
  macro(CUDA_SOURCE fileName)
    # nothing to do, leave it to cuda to compile this.
  endmacro()
else()
  # we don't actually _have_ cuda - tell a regular C++ compiler to
  # compile this source, and hope it has nothing in it that wouldn't
  # work on off-the-shelf c++
  macro(CUDA_SOURCE fileName)
    if (WIN32)
      set_source_files_properties(${fileName}
        PROPERTIES
        LANGUAGE CXX
        )
      
    else()
      set_source_files_properties(${fileName}
        PROPERTIES
        LANGUAGE CXX
        COMPILE_OPTIONS   "-xc++"
        )
    endif()
  endmacro()
endif()
# ==================================================================

CUDA_SOURCE(kernels/traceRays.cu)
CUDA_SOURCE(kernels/generateRays.cu)
CUDA_SOURCE(kernels/shadeRays.cu)
CUDA_SOURCE(geometry/Triangles.dev.cu)
CUDA_SOURCE(geometry/Spheres.dev.cu)
CUDA_SOURCE(geometry/Capsules.dev.cu)
CUDA_SOURCE(Camera.cu)
CUDA_SOURCE(fb/FrameBuffer.cu)
CUDA_SOURCE(fb/TiledFB.cu)
CUDA_SOURCE(fb/LocalFB.cu)
CUDA_SOURCE(light/DirLight.cu)
CUDA_SOURCE(light/PointLight.cu)
CUDA_SOURCE(light/QuadLight.cu)
CUDA_SOURCE(light/EnvMap.cu)
CUDA_SOURCE(kernels/shadeRays.cu)
CUDA_SOURCE(volume/MCGrid.cu)
CUDA_SOURCE(volume/StructuredData.cu)
CUDA_SOURCE(volume/StructuredData.dev.cu)
CUDA_SOURCE(umesh/common/UMeshField.cu)
CUDA_SOURCE(umesh/mc/UMeshCUBQLSampler.cu)
CUDA_SOURCE(umesh/os/RTXObjectSpace.cu)
CUDA_SOURCE(umesh/os/AWT.cu)
CUDA_SOURCE(amr/BlockStructuredField.cu)
CUDA_SOURCE(amr/BlockStructuredCUBQLSampler.cu)

list(APPEND BARNEY_DEPENDS
  # base config and rtcore - these go into barney, one way or another
  $<BUILD_INTERFACE:barney_config>
  $<BUILD_INTERFACE:barney_rtc_common>
  $<BUILD_INTERFACE:barney_rtc_optix>
  $<BUILD_INTERFACE:barney_rtc_cuda>
  $<BUILD_INTERFACE:barney_rtc_cuda_common>
  $<BUILD_INTERFACE:barney_rtc_embree>
  
  $<BUILD_INTERFACE:cuBQL_cpu_float3_static>
  )

if (BARNEY_HAVE_CUDA)
  list(APPEND BARNEY_DEPENDS
    $<BUILD_INTERFACE:cuBQL_cuda_float3_static>
    )
endif()
if (BARNEY_BACKEND_OPTIX)
  # optix embedded ptx code strings for the precompiled kernels. these
  # only go into barney if optix mode is enabled (because otherwise we
  # cannot even compile them)
  list(APPEND BARNEY_DEPENDS
    $<BUILD_INTERFACE:$<TARGET_OBJECTS:barney-triangles-ptx>>
    $<BUILD_INTERFACE:$<TARGET_OBJECTS:barney-capsules-ptx>>
    $<BUILD_INTERFACE:$<TARGET_OBJECTS:barney-spheres-ptx>>
    #
    $<BUILD_INTERFACE:$<TARGET_OBJECTS:barney-StructuredData-ptx>>
    $<BUILD_INTERFACE:$<TARGET_OBJECTS:barney-UMeshMC-ptx>>
    $<BUILD_INTERFACE:$<TARGET_OBJECTS:barney-AWT-ptx>>
    #
    $<BUILD_INTERFACE:$<TARGET_OBJECTS:barney-traceRays-ptx>>
    )
endif()

if (BARNEY_HAVE_CUDA)
        set_source_files_properties(Context.cpp
        PROPERTIES
        LANGUAGE CUDA
        )
endif()



if (BARNEY_MPI)
  
  # ------------------------------------------------------------------
  # SHARED version of barney_mpi
  # ------------------------------------------------------------------
  add_library(barney_mpi SHARED
    ${BARNEY_SOURCES}
    common/MPIWrappers.h
    common/MPIWrappers.cpp
    MPIContext.h
    MPIContext.cpp
    fb/DistFB.h
    fb/DistFB.cpp
    )
  set_target_properties(barney_mpi
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
    )
  target_link_libraries(barney_mpi PUBLIC
    ${BARNEY_DEPENDS}
    MPI::MPI_CXX
    )
  target_compile_definitions(barney_mpi PUBLIC -DBARNEY_MPI=1)
  if (APPLE)
    set_target_properties(barney_mpi PROPERTIES INSTALL_RPATH "$loader_path")
  else()
    set_target_properties(barney_mpi PROPERTIES INSTALL_RPATH "$ORIGIN")
  endif()


endif()
endif()


