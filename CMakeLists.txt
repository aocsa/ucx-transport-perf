cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(utecdf C CXX)


set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# - conda environment -----------------------------------------------------------------------------
if ("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(PREFIX $ENV{PREFIX})
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{BUILD_PREFIX};$ENV{PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    message(STATUS "Conda build detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
elseif (DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(PREFIX $ENV{CONDA_PREFIX})
    message(STATUS "Conda environment detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")

    include_directories(
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/include
            $ENV{CONDA_PREFIX}/include $ENV{PREFIX}/include)
    link_directories($ENV{CONDA_PREFIX}/lib)
endif ()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQURIED ON)

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CTest)

find_package(Threads)
find_package(GTest REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog          REQUIRED)
enable_testing ()
find_package(Threads)
include(ConfigureUCX)

if (GTEST_FOUND)
    enable_testing ()
    add_custom_target (coverage
            COMMAND lcov -c -d ${CMAKE_BINARY_DIR}/src -o coverage.info
            COMMAND lcov -r coverage.info '/usr*' '*boost*' '*build*' -o coverage.info
            COMMAND genhtml coverage.info -o ${CMAKE_BINARY_DIR}/coverage-html)
endif ()

function (blazingdb_artifact)
    set (OPTIONS OPTIONAL VIRTUAL)
    set (ONE_VALUE_ARGS PREFIX TARGET)
    set (MULTI_VALUE_ARGS SOURCES INCLUDES LIBRARIES PUBLIC_HEADERS TESTS)
    cmake_parse_arguments (ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})
    if (ARG_UNPARSED_ARGUMENTS)
        message (SEND_ERROR
                "Error: unrecognized arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif ()

    if (${ARG_VIRTUAL})
        set (TYPE OBJECT)
    endif ()


    if (ARG_PUBLIC_HEADERS)
        message (STATUS "Log: public headers ${ARG_PUBLIC_HEADERS}")
        install(DIRECTORY ${ARG_PUBLIC_HEADERS} DESTINATION include)
    endif ()

    add_library (${ARG_TARGET} ${TYPE} ${ARG_SOURCES})

    if (ARG_LIBRARIES)
        target_link_libraries (${ARG_TARGET} ${ARG_LIBRARIES})
    endif ()

    if (ARG_INCLUDES)
        target_include_directories (${ARG_TARGET} ${ARG_INCLUDES})
    endif ()

    if (NOT ARG_PREFIX)
        set (ARG_PREFIX "")
    endif ()

    install (TARGETS ${ARG_TARGET}
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)


    if (GTEST_FOUND AND ARG_TESTS)
        message (STATUS "Log: unit test ${ARG_TARGET}-gtest")

        add_executable (${ARG_TARGET}-gtest ${ARG_TESTS})
        target_include_directories (${ARG_TARGET} PUBLIC src)
        target_compile_options (${ARG_TARGET} PUBLIC -ggdb -O0)
        target_link_libraries (${ARG_TARGET}-gtest
                ${ARG_TARGET}
                GTest::GTest
                $ENV{CONDA_PREFIX}/lib/libgmock_main.a
                $ENV{CONDA_PREFIX}/lib/libgmock.a )

        add_custom_target (${ARG_TARGET}-gtest-run
                COMMAND $<TARGET_FILE:${ARG_TARGET}-gtest>)

        add_test(NAME ${ARG_TARGET}-gtest
                COMMAND
                ${CMAKE_BINARY_DIR}/${ARG_TARGET}-gtest
                )
        add_dependencies (coverage ${ARG_TARGET}-gtest-run)
    endif ()

endfunction ()

blazingdb_artifact (
        TARGET
        ucx_perf

        INCLUDES PUBLIC
        $ENV{PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include
        $ENV{CONDA_PREFIX}/include
        /usr/local/cuda-10.0/include

        LIBRARIES
        Threads::Threads
        zmq
        libboost_filesystem.a
        libboost_system.a
        libboost_regex.a
        fmt::fmt
        gflags
        spdlog

        cudart
        cuda
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_NVRTC_LIBRARY}
        ${CUDA_NVTX_LIBRARY}
        $ENV{CONDA_PREFIX}/lib64/libgdrapi.so

        ${GDR_STATIC_LIB}
        ${UCX_UCT_SHARED_LIB}
        ${UCX_UCS_SHARED_LIB}
        ${UCX_UCM_SHARED_LIB}

        PREFIX
        blazingdb

        PUBLIC_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/

        SOURCES
            src/socket.cpp
            src/blazingdb/uc/Capabilities.cpp
            src/blazingdb/uc/Context.cpp
            src/blazingdb/uc/ContextHelpers.cpp
            src/blazingdb/uc/internal/agents/AddressableAgent.cpp
            src/blazingdb/uc/internal/agents/ManagedAgent.cpp
            src/blazingdb/uc/internal/agents/ViewAgent.cpp
            src/blazingdb/uc/internal/buffers/LinkerBuffer.cpp
            src/blazingdb/uc/internal/buffers/RemoteBuffer.cpp
            src/blazingdb/uc/internal/buffers/ViewBuffer.cpp
            src/blazingdb/uc/internal/buffers/records/RemotableRecord.cpp
            src/blazingdb/uc/internal/buffers/transports/ZCopyTransport.cpp
            src/blazingdb/uc/internal/ManagedContext.cpp
            src/blazingdb/uc/internal/ViewContext.cpp
)
message (STATUS "***GDR_STATIC_LIB  ${GDR_STATIC_LIB}")

add_executable(ucx_server server.cpp)
target_include_directories (ucx_server PUBLIC src)
target_link_libraries (ucx_server ucx_perf)

add_executable(ucx_client client.cpp)
target_include_directories (ucx_client PUBLIC src)
target_link_libraries (ucx_client ucx_perf)


add_executable(ucx_benchmark bench.cpp)
target_include_directories (ucx_benchmark PUBLIC src)
target_link_libraries (ucx_benchmark ucx_perf)
