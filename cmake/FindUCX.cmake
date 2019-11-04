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

# Tries to find UCX headers and libraries.
#
# Usage of this module as follows:
#
#  find_package(UCX)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  UCX_HOME - When set, this path is inspected instead of standard library
#                locations as the root of the UCX installation.
#                The environment variable UCX_HOME overrides this variable.
#
# This module defines
#  UCX_INCLUDE_DIR, directory containing headers
#  UCX_LIBS, directory containing ucx libraries
#  UCX_STATIC_LIB, path to libucx.a
#  UCX_SHARED_LIB, path to libucx's shared library
#  UCX_FOUND, whether ucx has been found

if( NOT "${UCX_HOME}" STREQUAL "")
    file( TO_CMAKE_PATH "${UCX_HOME}" _native_path )
    list( APPEND _ucx_roots ${_native_path} )
elseif ( UCX_HOME )
    list( APPEND _ucx_roots ${UCX_HOME} )
endif()

# BEGIN check UCX UCM headers

message(STATUS "UCX_HOME: ${UCX_HOME}")
find_path(UCX_INCLUDE_DIR ucm/api/ucm.h HINTS
  ${_ucx_roots}
  NO_DEFAULT_PATH
  PATH_SUFFIXES "include")

if (NOT UCX_INCLUDE_DIR)
  message(FATAL_ERROR "Could not found UCM headers: ${_ucx_roots}/include/")
endif()

# END check UCX UCM headers

# BEGIN check UCX UCP headers

message(STATUS "UCX_HOME: ${UCX_HOME}")
find_path(UCX_INCLUDE_DIR ucp/api/ucp.h HINTS
  ${_ucx_roots}
  NO_DEFAULT_PATH
  PATH_SUFFIXES "include")

if (NOT UCX_INCLUDE_DIR)
  message(FATAL_ERROR "Could not found UCP headers: ${_ucx_roots}/include/")
endif()

# END check UCX UCP headers

# BEGIN check UCX UCS headers

message(STATUS "UCX_HOME: ${UCX_HOME}")
find_path(UCX_INCLUDE_DIR ucs/config/types.h HINTS
  ${_ucx_roots}
  NO_DEFAULT_PATH
  PATH_SUFFIXES "include")

if (NOT UCX_INCLUDE_DIR)
  message(FATAL_ERROR "Could not found UCP headers: ${_ucx_roots}/include/")
endif()

# END check UCX UCS headers

# BEGIN check UCX UCT headers

message(STATUS "UCX_HOME: ${UCX_HOME}")
find_path(UCX_INCLUDE_DIR uct/api/uct.h HINTS
  ${_ucx_roots}
  NO_DEFAULT_PATH
  PATH_SUFFIXES "include")

if (NOT UCX_INCLUDE_DIR)
  message(FATAL_ERROR "Could not found UCP headers: ${_ucx_roots}/include/")
endif()

# END check UCX UCT headers

find_library( UCX_LIBRARIES NAMES ucp PATHS
  ${_ucx_roots}
  NO_DEFAULT_PATH
  PATH_SUFFIXES "lib")

if (UCX_INCLUDE_DIR AND (PARQUET_MINIMAL_DEPENDENCY OR UCX_LIBRARIES))
  set(UCX_FOUND TRUE)
  get_filename_component( UCX_LIBS ${UCX_LIBRARIES} PATH )
  set(UCX_HEADER_NAME ucp/api/ucp.h)
  set(UCX_HEADER ${UCX_INCLUDE_DIR}/${UCX_HEADER_NAME})

  set(UCX_LIB_NAME ucm)
  set(UCX_UCM_STATIC_LIB ${UCX_LIBS}/${CMAKE_STATIC_LIBRARY_PREFIX}${UCX_LIB_NAME}${UCX_MSVC_STATIC_LIB_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(UCX_UCM_SHARED_LIB ${UCX_LIBS}/${CMAKE_SHARED_LIBRARY_PREFIX}${UCX_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX})

  set(UCX_LIB_NAME ucp)
  set(UCX_UCP_STATIC_LIB ${UCX_LIBS}/${CMAKE_STATIC_LIBRARY_PREFIX}${UCX_LIB_NAME}${UCX_MSVC_STATIC_LIB_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(UCX_UCP_SHARED_LIB ${UCX_LIBS}/${CMAKE_SHARED_LIBRARY_PREFIX}${UCX_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX})

  set(UCX_LIB_NAME ucs)
  set(UCX_UCS_STATIC_LIB ${UCX_LIBS}/${CMAKE_STATIC_LIBRARY_PREFIX}${UCX_LIB_NAME}${UCX_MSVC_STATIC_LIB_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(UCX_UCS_SHARED_LIB ${UCX_LIBS}/${CMAKE_SHARED_LIBRARY_PREFIX}${UCX_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX})

  set(UCX_LIB_NAME uct)
  set(UCX_UCT_STATIC_LIB ${UCX_LIBS}/${CMAKE_STATIC_LIBRARY_PREFIX}${UCX_LIB_NAME}${UCX_MSVC_STATIC_LIB_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(UCX_UCT_SHARED_LIB ${UCX_LIBS}/${CMAKE_SHARED_LIBRARY_PREFIX}${UCX_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX})
else ()
  set(UCX_FOUND FALSE)
endif ()

if (UCX_FOUND)
  if (NOT UCX_FIND_QUIETLY)
    if (PARQUET_MINIMAL_DEPENDENCY)
      message(STATUS "Found the UCX header: ${UCX_HEADER}")
    else ()
      message(STATUS "Found the UCX library: ${UCX_LIBRARIES}")
    endif ()
  endif ()
else ()
  if (NOT UCX_FIND_QUIETLY)
    set(UCX_ERR_MSG "Could not find the UCX library. Looked in ")
    if ( _ucx_roots )
      set(UCX_ERR_MSG "${UCX_ERR_MSG} in ${_ucx_roots}.")
    else ()
      set(UCX_ERR_MSG "${UCX_ERR_MSG} system search paths.")
    endif ()
    if (UCX_FIND_REQUIRED)
      message(FATAL_ERROR "${UCX_ERR_MSG}")
    else (UCX_FIND_REQUIRED)
      message(STATUS "${UCX_ERR_MSG}")
    endif (UCX_FIND_REQUIRED)
  endif ()
endif ()

mark_as_advanced(
  UCX_INCLUDE_DIR
  UCX_LIBS
  UCX_LIBRARIES

  UCX_UCM_STATIC_LIB
  UCX_UCM_SHARED_LIB

  UCX_UCP_STATIC_LIB
  UCX_UCP_SHARED_LIB

  UCX_UCS_STATIC_LIB
  UCX_UCS_SHARED_LIB

  UCX_UCT_STATIC_LIB
  UCX_UCT_SHARED_LIB
)
