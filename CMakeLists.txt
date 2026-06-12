############################## Set up dependencies
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Numstore REQUIRED)
find_package(Numpy REQUIRED)

############################## Compile 
Python_add_library(_pynumstore MODULE WITH_SOABI pynumstore.c ${NSPY_SOURCES})
target_include_directories(_pynumstore
    PRIVATE "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/c/include/"
            "${CMAKE_CURRENT_SOURCE_DIR}/c/src/"
)

# Include Numpy includes seperately to ignore compiler flags
target_include_directories(_pynumstore
    SYSTEM PRIVATE  "${NumPy_INCLUDE}"
)

target_link_libraries(_pynumstore PRIVATE numstore)
install(TARGETS _pynumstore DESTINATION pynumstore)
