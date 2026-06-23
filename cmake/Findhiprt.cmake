# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# \author Jeff Daily <jeff.daily@amd.com>

# Locate the AMD HIPRT SDK (https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT).
# Mirrors how barney finds OptiX/OIDN: a discovered dependency, never vendored,
# pointed at by hiprt_ROOT (or the HIPRT_PATH environment variable). HIPRT
# supplies the ray-tracing BVH build + traversal used by the rtcore/hiprt
# backend; on a GPU without hardware RT units (e.g. gfx90a) it falls back to a
# correct software BVH traversal.
#
# Sets:
#   hiprt_FOUND
#   hiprt::hiprt   imported target (include dir + libhiprt)
#   hiprt_INCLUDE_DIR
#   hiprt_LIBRARY

find_path(hiprt_INCLUDE_DIR
  NAMES hiprt/hiprt.h
  HINTS
    ${hiprt_ROOT}
    $ENV{hiprt_ROOT}
    $ENV{HIPRT_PATH}
  )

# HIPRT names its shared library libhiprt<version>.so (e.g. libhiprt0300164.so);
# accept any so the consumer need not hardcode the exact build version.
find_library(hiprt_LIBRARY
  NAMES hiprt hiprt0300264 hiprt0300164 hiprt0300064
  HINTS
    ${hiprt_ROOT}
    $ENV{hiprt_ROOT}
    $ENV{HIPRT_PATH}
  PATH_SUFFIXES
    dist/bin/Release
    dist/bin/Debug
    bin
    lib
  )

if (NOT hiprt_LIBRARY AND hiprt_INCLUDE_DIR)
  file(GLOB _hiprt_candidates
    "${hiprt_INCLUDE_DIR}/dist/bin/Release/libhiprt*.so"
    "${hiprt_INCLUDE_DIR}/dist/bin/Debug/libhiprt*.so"
    "${hiprt_INCLUDE_DIR}/lib/libhiprt*.so"
    "${hiprt_INCLUDE_DIR}/bin/libhiprt*.so")
  if (_hiprt_candidates)
    list(GET _hiprt_candidates 0 hiprt_LIBRARY)
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hiprt
  REQUIRED_VARS hiprt_INCLUDE_DIR hiprt_LIBRARY)

if (hiprt_FOUND AND NOT TARGET hiprt::hiprt)
  add_library(hiprt::hiprt UNKNOWN IMPORTED)
  set_target_properties(hiprt::hiprt PROPERTIES
    IMPORTED_LOCATION "${hiprt_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${hiprt_INCLUDE_DIR}")
endif()

mark_as_advanced(hiprt_INCLUDE_DIR hiprt_LIBRARY)
