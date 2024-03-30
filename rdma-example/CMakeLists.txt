# Author : Animesh Trivedi
function(llama_build_executable source)
    get_filename_component(TEST_TARGET ${source} NAME_WE)
    add_executable(${TEST_TARGET} ${source})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE llama common)
endfunction()

function(llama_test_executable name source)
    get_filename_component(TEST_TARGET ${source} NAME_WE)
    add_test(NAME ${name} COMMAND $<TARGET_FILE:${TEST_TARGET}> ${ARGN})
endfunction()

function(llama_build_and_test_executable source)
    get_filename_component(TEST_TARGET ${source} NAME_WE)
    add_executable(${TEST_TARGET} ${source} ${PROJECT_SOURCE_DIR}/rdma_common.cpp)
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE llama common )
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}> ${ARGN})
endfunction()

cmake_minimum_required (VERSION 2.6)

project (rdma-example)
if (ACN_RDMA_DEBUG)
    add_compile_definitions(ACN_RDMA_DEBUG)
endif()
set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/rdma-example/src)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/rdma-example/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Release) # or debug if you need to use gdb on it.

# Some how g++ has better performance than clang++. Of course I don't know all flags for clang++.
set(CMAKE_CXX_COMPILER g++) # or clang++
#set(CMAKE_CXX_COMPILER clang++) # or clang++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Ofast -ffast-math -funroll-loops -march=native") # get crazy here
#add_compile_options("-std=c++17")
#set(CMAKE_CXX_FLAGS "-O0  -ggdb")

find_library(IBVERBS_LIBRARY ibverbs HINTS /home/atr/local/lib)
find_library(RDMACM_LIBRARY rdmacm HINTS /home/atr/local/lib)

link_libraries(pthread ${IBVERBS_LIBRARY} ${RDMACM_LIBRARY})

include_directories("${PROJECT_SOURCE_DIR}" "/home/atr/local/include/")
include_directories("/home/yumj/powerinfer" "/home/atr/local/include/")

# add_executable(rdma_server ${PROJECT_SOURCE_DIR}/rdma_common.c ${PROJECT_SOURCE_DIR}/rdma_server.c)
# add_executable(rdma_client ${PROJECT_SOURCE_DIR}/rdma_common.c ${PROJECT_SOURCE_DIR}/rdma_client.c)
# add_executable(rdma_server_LLM ${PROJECT_SOURCE_DIR}/rdma_common.c ${PROJECT_SOURCE_DIR}/rdma_server_LLM.c)
# add_executable(rdma_client_LLM ${PROJECT_SOURCE_DIR}/rdma_common.c ${PROJECT_SOURCE_DIR}/rdma_client_LLM.c)
# llama_build_and_test_executable(${PROJECT_SOURCE_DIR}/rdma_client_LLM.c)
llama_build_and_test_executable(${PROJECT_SOURCE_DIR}/rdma_server_LLM.cpp)
llama_build_and_test_executable(${PROJECT_SOURCE_DIR}/rdma_server_LLM_API.cpp)
llama_build_and_test_executable(${PROJECT_SOURCE_DIR}/rdma_client_LLM.cpp)
llama_build_and_test_executable(${PROJECT_SOURCE_DIR}/rdma_client_LLM_API.cpp)

# llama_build_and_test_executable(${PROJECT_SOURCE_DIR}/rdma_client.c)
# llama_build_and_test_executable(${PROJECT_SOURCE_DIR}/rdma_server.c)

set(TARGET rdma_common)

add_library(${TARGET} 
    ${PROJECT_SOURCE_DIR}/rdma_common.cpp
    ${PROJECT_SOURCE_DIR}/rdma_common.h
    )

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PUBLIC .)
target_compile_features(${TARGET} PUBLIC cxx_std_11)
target_link_libraries(${TARGET} PRIVATE llama build_info)
