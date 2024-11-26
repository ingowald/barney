## Copyright 2023 Ingo Wald
## SPDX-License-Identifier: Apache-2.0

option(BARNEY_BUILD_ANARI "Build ANARI device" OFF)
if (NOT BARNEY_BUILD_ANARI)
  return()
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(BANARI_STATIC ON)
if (BANARI_STATIC)
  set(BANARI_LIBRARY_TYPE STATIC)
  set(ANARI_LIBRARY_NAME anari::anari_static)
  set(BARNEY_LIBRARY_SUFFIX _static)
else()
  set(BANARI_LIBRARY_TYPE SHARED)
  set(ANARI_LIBRARY_NAME anari::anari)
  set(BARNEY_LIBRARY_SUFFIX "")
endif()
enable_language(CUDA)

find_package(anari 0.10.0 COMPONENTS code_gen)

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

add_library(banari-config INTERFACE
  )

add_library(anari_library_barney SHARED
  ${BANARI_SOURCES}
)
add_library(anari_library_barney_static OBJECT
  ${BANARI_SOURCES}
  )


#set_source_files_properties(
#  Frame.cpp
#  PROPERTIES
#    COMPILE_FLAGS "--extended-lambda --expt-relaxed-constexpr"
#    LANGUAGE CUDA
#)

#if (TARGET barney_mpi)
#  project_link_libraries(PRIVATE anari::helium barney_mpi${BARNEY_LIBRARY_SUFFIX})
#else()
#  set(CMAKE_CXX_LINK_LIBRARY_USING_WHOLE_ARCHIVE 1)


target_link_libraries(anari_library_barney PRIVATE
  banari-config
  anari::helium
  #  "$<LINK_LIBRARY:WHOLE_ARCHIVE,barney${BARNEY_LIBRARY_SUFFIX}>"
#  barney-config
  barney
#  owl::owl
#  barney${BARNEY_LIBRARY_SUFFIX}
  )
target_link_libraries(anari_library_barney_static PUBLIC
  banari-config
  anari::helium
  #  "$<LINK_LIBRARY:WHOLE_ARCHIVE,barney${BARNEY_LIBRARY_SUFFIX}>"
#  barney-config
  barney_static
#  owl::owl
#  barney${BARNEY_LIBRARY_SUFFIX}
  )


target_include_directories(banari-config
  INTERFACE#PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

include(GenerateExportHeader)
generate_export_header(anari_library_barney #${PROJECT_NAME}
  EXPORT_MACRO_NAME "BARNEY_LIBRARY_INTERFACE"
)

## Installation ##

install(TARGETS anari_library_barney#${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

## Test app ##

project(anariTest LANGUAGES CXX)
project_add_executable(test/anariTest.cpp)
project_link_libraries(${ANARI_LIBRARY_NAME})
#project_link_libraries(anari::anari)
