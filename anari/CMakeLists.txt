## Copyright 2023 Ingo Wald
## SPDX-License-Identifier: Apache-2.0

option(BARNEY_BUILD_ANARI "Build ANARI device" OFF)
if (NOT BARNEY_BUILD_ANARI)
  return()
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(anari 0.12.0 COMPONENTS code_gen)
include(GNUInstallDirs)

## Code generation ##
anari_generate_queries(
  NAME barney
  PREFIX BarneyDevice
  CPP_NAMESPACE barney_device
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/barney_device.json
  )

## Device target ##

project(anari_library_barney LANGUAGES CXX)

set(BANARI_SOURCES
  BarneyDeviceQueries.cpp
  BarneyGlobalState.cpp
  Camera.cpp
  Device.cpp
  Frame.cpp
  Geometry.cpp
  Group.cpp
  Instance.cpp
  Library.cpp
  Light.cpp
  Material.cpp
  Object.cpp
  Renderer.cpp
  Sampler.cpp
  SpatialField.cpp
  Surface.cpp
  Volume.cpp
  World.cpp
  )

# helper 'interface-'library that contains common settings for both
# static and shared anari
add_library(banari-config INTERFACE
  )
set_target_properties(banari-config
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  )
set_target_properties(banari-config
  PROPERTIES
  CUDA_ARCHITECTURES all-major
  )

target_include_directories(banari-config
  INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# default shared-linkage for anari loader library to load when barney
# device is requested
add_library(anari_library_barney SHARED
  ${BANARI_SOURCES}
  )

# static library that contains a dedicated 'create'-function that
# generates _only_ a banari device 
add_library(anari_library_barney_static
  #OBJECT
  STATIC
  ${BANARI_SOURCES}
  )

target_link_libraries(anari_library_barney PRIVATE
  banari-config
  anari::helium
  barney
  )
target_link_libraries(anari_library_barney_static PUBLIC
  banari-config
#  anari::helium_static
  anari::helium
  barney_static
  )
set_target_properties(anari_library_barney
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  )
set_target_properties(anari_library_barney_static
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  )



include(GenerateExportHeader)
generate_export_header(anari_library_barney
  EXPORT_MACRO_NAME "BARNEY_LIBRARY_INTERFACE"
  )
generate_export_header(anari_library_barney_static
  EXPORT_MACRO_NAME "BARNEY_LIBRARY_INTERFACE"
  )

## Installation ##

install(TARGETS anari_library_barney
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

