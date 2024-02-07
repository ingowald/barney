# ======================================================================== #
# Copyright 2023-2024 Ingo Wald                                            #
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

enable_testing()

# simple test that just - in a loop - keeps on creating and destroying
# barney contexts, to check for ramp-up/ramp-down errors or context
# memory leaks
add_executable(bnTest_contextCreateDestroy contextCreateDestroy.cpp)
target_link_libraries(bnTest_contextCreateDestroy PUBLIC barney)
add_test(NAME bnTest_contextCreateDestroy COMMAND bnTest_contextCreateDestroy)



# checks that we can create and release 'some stuff' without crashing
# or leaking
add_executable(bnTest_createRelease createRelease.cpp)
target_link_libraries(bnTest_createRelease PUBLIC barney)
add_test(NAME bnTest_createRelease COMMAND bnTest_createRelease)


# in a loop, this creates a model, builds that (empty) model, and
# releases it. can be used to check for memory leaks in bnBuild()
add_executable(bnTest_createBuildAndRelease createBuildAndRelease.cpp)
target_link_libraries(bnTest_createBuildAndRelease PUBLIC barney)
add_test(NAME bnTest_createBuildAndRelease COMMAND bnTest_createBuildAndRelease)


# in a loop, this creates a frame buffer (of random size and randomly
# with or without depth buffer), then releases it
add_executable(bnTest_createReleaseFB createReleaseFB.cpp)
target_link_libraries(bnTest_createReleaseFB PUBLIC barney)
add_test(NAME bnTest_createReleaseFB COMMAND bnTest_createReleaseFB)


# in a loop, this creates a mesh of trianlges, and frees it
add_executable(bnTest_createReleaseTriangles createReleaseTriangles.cpp)
target_link_libraries(bnTest_createReleaseTriangles PUBLIC barney)
add_test(NAME bnTest_createReleaseTriangles COMMAND bnTest_createReleaseTriangles)


# in a loop, this creates a mesh of trianlges, and frees it
add_executable(bnTest_createReleaseAndBuildTrianglesGroup createReleaseAndBuildTrianglesGroup.cpp)
target_link_libraries(bnTest_createReleaseAndBuildTrianglesGroup PUBLIC barney)
add_test(NAME bnTest_createReleaseAndBuildTrianglesGroup COMMAND bnTest_createReleaseAndBuildTrianglesGroup)

