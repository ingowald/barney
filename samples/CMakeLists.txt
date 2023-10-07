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

if (NOT (TARGET miniScene))
  add_subdirectory(../../miniScene ext_miniScene EXCLUDE_FROM_ALL)
endif()

add_executable(bnMpiStrawman
  strawman.cpp)
target_link_libraries(bnMpiStrawman PUBLIC
  barney_mpi
  miniScene)
target_compile_definitions(bnMpiStrawman PUBLIC -DSM_USE_MPI=1)


add_executable(bnLocalStrawman
  strawman.cpp)
target_link_libraries(bnLocalStrawman PUBLIC
  barney
  miniScene)
target_compile_definitions(bnLocalStrawman PUBLIC -DSM_NUM_LOCAL_DATA=1)

add_executable(bnMultiStrawman
  strawman.cpp)
target_link_libraries(bnMultiStrawman PUBLIC
  barney
  miniScene)
target_compile_definitions(bnMultiStrawman PUBLIC -DSM_NUM_LOCAL_DATA=2)
