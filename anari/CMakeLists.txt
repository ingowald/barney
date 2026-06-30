# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# "backend" variable gets set by parent dir, and is one of either
# 'cpu', 'cuda', 'optix', 'hip' or 'hiprt'
set(BANARI_NS banari_${backend})
set(BARNEY_DEVICE_NAME barney_${backend})

message("configuring library entrypoint, BARNEY_DEVICE_NAME = ${BARNEY_DEVICE_NAME}")
configure_file(
  ${PROJECT_SOURCE_DIR}/anari/Library_entryPoint.h.in
  Library_entryPoint.h
)

set(BARNEY_INSTALL_TARGETS)
set(BANARI_SOURCES
  Array.cpp
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
  Sampler.cpp
  SpatialField.cpp
  Surface.cpp
  Volume.cpp
  World.cpp
  # -----------
  light/Light.h
  light/Light.cpp
  light/HDRI.h
  light/HDRI.cpp
  light/Directional.h
  light/Directional.cpp
  light/Point.h
  light/Point.cpp

  # 'force' a single cuda file into the library, so cmake will
  # properly recognize this as requiring linking with cuda.
  dummy.cu
)
rtc_configure_source(dummy.cu)

if (NOT BARNEY_AUTO_GENERATE_QUERIES)
  LIST(APPEND BANARI_SOURCES generated/anari_library_barney_queries.cpp)
endif()


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
add_library(anari_library_${BARNEY_DEVICE_NAME} SHARED ${BANARI_SOURCES})
target_link_libraries(anari_library_${BARNEY_DEVICE_NAME} PRIVATE
  $<BUILD_INTERFACE:banari-config>
  ${BARNEY_DEVICE_NAME}_static
  anari::helium
)
rtc_library_properties(anari_library_${BARNEY_DEVICE_NAME})
  target_link_options(anari_library_${BARNEY_DEVICE_NAME} PRIVATE
  $<$<NOT:$<PLATFORM_ID:Darwin>>:-Wl,--export-dynamic>
)

if (BARNEY_MPI)
  add_library(anari_library_${BARNEY_DEVICE_NAME}_mpi SHARED ${BANARI_SOURCES})
  target_link_libraries(anari_library_${BARNEY_DEVICE_NAME}_mpi PRIVATE
    $<BUILD_INTERFACE:banari-config>
    ${BARNEY_DEVICE_NAME}_mpi_static
    anari::helium
  )
  rtc_library_properties(anari_library_${BARNEY_DEVICE_NAME}_mpi)
  set(BARNEY_INSTALL_TARGETS
    anari_library_${BARNEY_DEVICE_NAME}
    anari_library_${BARNEY_DEVICE_NAME}_mpi
  )
else()
  set(BARNEY_INSTALL_TARGETS
    anari_library_${BARNEY_DEVICE_NAME}
  )
endif()

## Installation ##

if (APPLE)
  set_target_properties(anari_library_${BARNEY_DEVICE_NAME} PROPERTIES INSTALL_RPATH "$loader_path")
else()
  set_target_properties(anari_library_${BARNEY_DEVICE_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

file(GLOB DEVICE_JSONS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.json")
foreach(file ${DEVICE_JSONS})
  file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file} INPUT ${file})
endforeach()

if (BARNEY_AUTO_GENERATE_QUERIES)
  anari_generate_queries(
    DEVICE_TARGET anari_library_${BARNEY_DEVICE_NAME}
    CPP_NAMESPACE ${BANARI_NS}
    JSON_ROOT_LOCATION ${CMAKE_CURRENT_BINARY_DIR}
    JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_BINARY_DIR}/barney_device.json
    JSON_EXTENSION_FILES ${DEVICE_JSONS}
  )
endif()

if (BARNEY_MPI)
  if (BARNEY_AUTO_GENERATE_QUERIES)
    anari_generate_queries(
      DEVICE_TARGET anari_library_${BARNEY_DEVICE_NAME}_mpi
      CPP_NAMESPACE ${BANARI_NS}
      JSON_ROOT_LOCATION ${CMAKE_CURRENT_BINARY_DIR}
      JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_BINARY_DIR}/barney_device.json
      JSON_EXTENSION_FILES ${DEVICE_JSONS}
    )
  endif()
endif()

install(
  TARGETS
  ${BARNEY_INSTALL_TARGETS}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  NAMELINK_SKIP
  # on Windows put the dlls into bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  # ... and the import lib into the devel package
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
