# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------------
# optix backend: instantiates all the barney classes,Data device
# programs, etcpp - except the api itself - in a optix
# configuration.
# ------------------------------------------------------------------

enable_language(CUDA)

macro(rtc_library_properties lib)
endmacro()

macro(rtc_build_device_sources libname)
  add_library(${libname} STATIC ${ARGN})
  rtc_library_properties(${libname})
endmacro()

macro(rtc_configure_source)
  foreach(src ${ARGN})
    get_filename_component(ext "${src}" EXT)
    if (ext STREQUAL ".cu")
      set_source_files_properties(${src} PROPERTIES
        LANGUAGE CUDA
      )
    endif()
  endforeach()
endmacro()

