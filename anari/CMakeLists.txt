## Copyright 2023 Ingo Wald
## SPDX-License-Identifier: Apache-2.0

option(BARNEY_BUILD_ANARI "Build ANARI device" ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(anari 0.13.0 COMPONENTS code_gen)
if (NOT anari_FOUND)
  message("barney: ANARI SDK not found - skipping this")
  return()
endif()

include(GNUInstallDirs)

## Device target ##

if (BARNEY_HAVE_CUDA)
  project(anari_library_barney LANGUAGES CXX CUDA)
  set(DUMMY_CU dummy.cu)
else()
  project(anari_library_barney LANGUAGES CXX)
  set(DUMMY_CU)
endif()

set(BANARI_SOURCES
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
  ${DUMMY_CU}
  )



# helper 'interface-'library that contains common settings for both
# static and shared anari
add_library(banari-config INTERFACE)
target_link_libraries(banari-config INTERFACE barney_config)
set_target_properties(banari-config PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(banari-config
INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# ==================================================================
# default shared-linkage for anari loader library to load when barney
# device is requested
# ==================================================================
add_library(anari_library_barney SHARED ${BANARI_SOURCES})
target_link_libraries(anari_library_barney PRIVATE
  $<BUILD_INTERFACE:banari-config>
  barney_static
#  barney
  anari::helium
  anari::anari
)

if (BARNEY_MPI)
  add_library(anari_library_barney_mpi SHARED ${BANARI_SOURCES})
  target_link_libraries(anari_library_barney_mpi PRIVATE
    $<BUILD_INTERFACE:banari-config>
    barney_mpi_static
#    barney_mpi
    anari::helium
    anari::anari
  )
endif()


if (BARNEY_HAVE_CUDA)
  target_compile_definitions(anari_library_barney PRIVATE -DBANARI_HAVE_CUDA=1)
  if (BARNEY_MPI)
    target_compile_definitions(anari_library_barney_mpi PRIVATE -DBANARI_HAVE_CUDA=1)
  endif()
endif()


if (BARNEY_HAVE_CUDA)
  set_source_files_properties(World.cpp
    PROPERTIES
    LANGUAGE CUDA
  )
  set_target_properties(anari_library_barney
    PROPERTIES
    LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

#target_link_libraries(anari_library_barney PRIVATE )
# if(MSVC)
#     target_compile_options(anari_library_barney INTERFACE
#         "$<$<CONFIG:Debug>:/MTd>"
#         "$<$<CONFIG:Release>:/MT>"
#         "$<$<CONFIG:RelWithDebInfo>:/MT>"
#         "$<$<CONFIG:MinSizeRel>:/MT>"
#     )
# endif()
set_target_properties(anari_library_barney
  PROPERTIES
  CXX_VISIBILITY_PRESET  hidden
  CUDA_VISIBILITY_PRESET hidden
  CUDA_SEPARABLE_COMPILATION   ON
  POSITION_INDEPENDENT_CODE    ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CUDA_RESOLVE_DEVICE_SYMBOLS  ON
)

include(GenerateExportHeader)
generate_export_header(anari_library_barney
  EXPORT_MACRO_NAME "BARNEY_LIBRARY_INTERFACE"
)

## Installation ##

if (APPLE)
  set_target_properties(anari_library_barney PROPERTIES INSTALL_RPATH "$loader_path")
else()
  set_target_properties(anari_library_barney PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
if (OFF)
install(TARGETS# barney
  anari_library_barney
  EXPORT barney-config
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
endif()

file(GLOB DEVICE_JSONS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "visrtx_*.json")
list(APPEND DEVICE_JSONS "barney_device.json")

foreach(file ${DEVICE_JSONS})
  file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file} INPUT ${file})
endforeach()

anari_generate_queries(
  DEVICE_TARGET anari_library_barney
  CPP_NAMESPACE barney_device
  JSON_ROOT_LOCATION ${CMAKE_CURRENT_BINARY_DIR}
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_BINARY_DIR}/barney_device.json
  JSON_EXTENSION_FILES ${DEVICE_JSONS}
)
if (BARNEY_MPI)
  anari_generate_queries(
    DEVICE_TARGET anari_library_barney_mpi
    CPP_NAMESPACE barney_device
    JSON_ROOT_LOCATION ${CMAKE_CURRENT_BINARY_DIR}
    JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_BINARY_DIR}/barney_device.json
    JSON_EXTENSION_FILES ${DEVICE_JSONS}
  )
endif()
  
