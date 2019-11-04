#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

cmake_minimum_required(VERSION 3.12)

project(ucx-download NONE)

include(ExternalProject)

# WARNING it seems ucx doesn't support parallel builds, so don't use -jX args when calling make
ExternalProject_Add(ucx
        GIT_REPOSITORY    https://github.com/openucx/ucx.git
        GIT_TAG           v1.5.0
        SOURCE_DIR        "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/ucx-src"
        BUILD_IN_SOURCE   1
        INSTALL_DIR       "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/ucx-install"
        CONFIGURE_COMMAND ./autogen.sh COMMAND ./contrib/configure-devel --prefix=${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/ucx-install --with-cuda=$CUDA_HOME --enable-mt CPPFLAGS="-I//$CUDA_HOME/include"
        BUILD_COMMAND     ${CMAKE_MAKE_PROGRAM} install
        UPDATE_COMMAND    ""
        CMAKE_ARGS        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        )