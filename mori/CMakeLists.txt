# ======================================================================== #
# Copyright 2023-2023 Ingo Wald                                            #
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

# add_library(mori STATIC #SHARED
#   DeviceGroup.h
#   DeviceGroup.cpp
#   TiledFB.h
#   TiledFB.cu
#   Camera.h
#   Camera.cu
#   MoriContext.h
#   MoriContext.cu
#   Spheres.h
#   Spheres.cu
#   )


# target_link_libraries(mori
#   PUBLIC
#   owl
#   )
# target_link_libraries(mori
#   PUBLIC
#   mori-spheres-ptx
#   )

# target_include_directories(mori PUBLIC
#   ${CMAKE_CURRENT_LIST_DIR}/..
#   )
# set_target_properties(mori
#   PROPERTIES
#   CUDA_SEPARABLE_COMPILATION ON
#   POSITION_INDEPENDENT_CODE ON
#   CUDA_RESOLVE_DEVICE_SYMBOLS ON
#   CUDA_USE_STATIC_CUDA_RUNTIME ON
# )


