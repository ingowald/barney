## Copyright 2023 Ingo Wald
## SPDX-License-Identifier: Apache-2.0

option(BARNEY_BUILD_ANARI "Build ANARI device" OFF)
if (NOT BARNEY_BUILD_ANARI)
  return()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(anari 0.8.0 COMPONENTS code_gen)

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

project_add_library(SHARED
  BarneyDeviceQueries.cpp
  BarneyGlobalState.cpp
  Camera.cpp
  Device.cpp
  Frame.cpp
  Geometry.cpp
  Group.cpp
  Instance.cpp
  Library.cpp
  Material.cpp
  Object.cpp
  Renderer.cpp
  SpatialField.cpp
  Surface.cpp
  Volume.cpp
  World.cpp
)

project_link_libraries(PRIVATE anari::helium barney_mpi)
#project_link_libraries(PRIVATE anari::helium barney)

project_include_directories(
PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME}
  EXPORT_MACRO_NAME "BARNEY_LIBRARY_INTERFACE"
)

## Installation ##

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

## Test app ##

project(anariTest LANGUAGES CXX)
project_add_executable(test/anariTest.cpp)
project_link_libraries(anari::anari)
