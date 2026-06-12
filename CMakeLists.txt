################### Meta Information
cmake_minimum_required(VERSION 3.18)

## Read the version from version.txt
file(STRINGS "${CMAKE_SOURCE_DIR}/version.txt" NUMSTORE_VERSION LIMIT_COUNT 1)

project(pynumstore
    VERSION     "${NUMSTORE_VERSION}"
    DESCRIPTION "Python Bindings for Numstore"
    LANGUAGES   C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options(
        /std:c11
        /experimental:c11atomics
    )
endif()

############################## Set up dependencies
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Numstore REQUIRED)
find_package(Numpy REQUIRED)

############################## Compile 
Python_add_library(_pynumstore MODULE WITH_SOABI pynumstore.c ${NSPY_SOURCES})
target_include_directories(_pynumstore
    PRIVATE "${PROJECT_SOURCE_DIR}/thirdparty/numstore/include"
            "${PROJECT_SOURCE_DIR}/thirdparty/numstore/src"
)

# Include Numpy includes seperately to ignore compiler flags
target_include_directories(_pynumstore
    SYSTEM PRIVATE  "${NumPy_INCLUDE}"
)

target_link_libraries(_pynumstore PRIVATE numstore)
install(TARGETS _pynumstore DESTINATION pynumstore)
