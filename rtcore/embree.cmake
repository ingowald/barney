# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------------
# 'cpu' backend: instantiates all the barney classes, device
# programs, etcpp - except the api itself - in an embree configuration
# that'll run on the host.
# ------------------------------------------------------------------

# ==================================================================
message("enabling CPU backend (via embree)")

function(rtc_library_properties lib)
endfunction()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/embree buildDir_rtc_embree)

# macro(rtc_configure_source)
#   foreach(src ${ARGN})
#     get_filename_component(ext "${src}" EXT)
#     if (${ext} STREQUAL ".cu")
#       set_source_files_properties(${src} PROPERTIES
#         LANGUAGE ${BARNEY_DEVICE_LANGUAGE}
#         COMPILE_OPTIONS "--extended-lambda;-rdc=true"
#       )
#     endif()
#   endforeach()
# endmacro()


function(rtc_build_device_sources libname)
  message("rtc-embree: dev lib ${libname} adding device sourcess ${ARGN}")
  add_library(${libname} STATIC ${ARGN})
  rtc_configure_source(${ARGN})
  target_link_libraries(${libname} barney_rtc_embree)
  rtc_library_properties(${libname})
endfunction()

