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

if (BARNEY_MPI)
  target_compile_definitions(barney_config INTERFACE
    -DBARNEY_MPI=1
    )
endif()


add_library(barney_config_ptx INTERFACE)
target_link_libraries(barney_config_ptx
  INTERFACE
  barney_config
  )
target_compile_definitions(barney_config_ptx
  INTERFACE
  -DBARNEY_DEVICE_PROGRAM=1
  )


# ------------------------------------------------------------------
# barney itself
# ------------------------------------------------------------------
add_library(barney SHARED
  api/barney.cu
  )
target_link_libraries(barney PUBLIC
  $<BUILD_INTERFACE:barney_config>
  $<BUILD_INTERFACE:barney_rtc>
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
  api/barney.cu
  )
if (NOT BARNEY_HAVE_CUDA)
  set_source_files_properties(api/barney.cu PROPERTIES LANGUAGE CXX)
  if (NOT WIN32)
    set_source_files_properties(api/barney.cu PROPERTIES COMPILE_OPTIONS "-xc++")
  endif()
endif()

target_compile_definitions(barney_static PUBLIC
  -Dbarney_STATIC=1
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
  set_target_properties(barney_static PROPERTIES INSTALL_RPATH "$loader_path")
else()
  set_target_properties(barney_static PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()




set(HOST_SOURCES
  umesh/common/UMeshField.h
  umesh/common/UMeshField.cu
  umesh/mc/UMeshCUBQLSampler.h
  umesh/mc/UMeshCUBQLSampler.cu

#  umesh/os/AWT.cu
#  umesh/os/AWT.h
  
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

if (BARNEY_API)
  list(APPEND BARNEY_HOST_SOURCES
    fb/DistFB.h
    fb/DistFB.cpp
    MPIContext.h
    MPIContext.cpp
    )
endif()

set(DEVICE_PROGRAM_SOURCES
  geometry/Spheres.dev.cu
  geometry/Triangles.dev.cu
  geometry/Cylinders.dev.cu
  geometry/Capsules.dev.cu
  volume/StructuredData.dev.cu
  umesh/mc/UMeshMC.dev.cu
#  umesh/os/AWT.dev.cu
  amr/BlockStructuredMC.dev.cu
  kernels/traceRays.dev.cu
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
  set_target_properties(barney_optix
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION   ON
    POSITION_INDEPENDENT_CODE    ON
    CUDA_USE_STATIC_CUDA_RUNTIME ON
#    CUDA_RESOLVE_DEVICE_SYMBOLS  ON
    CXX_VISIBILITY_PRESET  default
    CUDA_VISIBILITY_PRESET default
    )
  target_link_libraries(barney_optix PUBLIC barney_config barney_rtc_optix)
  
  target_link_libraries(barney        PUBLIC
    $<BUILD_INTERFACE:barney_optix>
    $<BUILD_INTERFACE:cuBQL_cuda_float3_static>
    )
  target_link_libraries(barney_static PUBLIC
    $<BUILD_INTERFACE:barney_optix>
    $<BUILD_INTERFACE:cuBQL_cuda_float3_static>
    )
endif()

if (BARNEY_BACKEND_EMBREE)
  if (NOT BARNEY_HAVE_CUDA)
    foreach(src ${HOST_SOURCES} ${DEVICE_PROGRAM_SOURCES})
      get_filename_component(ext "${src}" EXT)
      if (("${ext}" STREQUAL ".cu") OR ("${ext}" STREQUAL ".dev.cu"))
	message("setting language of ${src} to CXX")
	set_source_files_properties(${src} PROPERTIES LANGUAGE CXX)
	if (NOT WIN32)
          set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "-xc++")
	endif()
      endif()
    endforeach()
  endif()
  
  add_library(barney_embree_host STATIC ${HOST_SOURCES})
  target_link_libraries(barney_embree_host PUBLIC barney_config barney_rtc_embree)
  
  add_library(barney_embree_programs STATIC ${DEVICE_PROGRAM_SOURCES})  
  target_link_libraries(barney_embree_programs PUBLIC barney_config barney_rtc_embree)
  target_compile_definitions(barney_embree_programs PRIVATE -DBARNEY_DEVICE_PROGRAM=1)

  target_link_libraries(barney PRIVATE $<BUILD_INTERFACE:barney_embree_host>)
  target_link_libraries(barney PRIVATE $<BUILD_INTERFACE:barney_embree_programs>)
  target_link_libraries(barney PUBLIC $<BUILD_INTERFACE:cuBQL_cpu_float3_static>)
  target_link_libraries(barney_static PRIVATE $<BUILD_INTERFACE:barney_embree_host>)
  target_link_libraries(barney_static PRIVATE $<BUILD_INTERFACE:barney_embree_programs>)
  target_link_libraries(barney_static PUBLIC $<BUILD_INTERFACE:cuBQL_cpu_float3_static>)
endif()
    













