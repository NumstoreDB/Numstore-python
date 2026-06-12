# FindNumpy.cmake
# Finds NumPy include directory via the active Python interpreter.
#
# Requires Python_EXECUTABLE to be set (e.g. from find_package(Python REQUIRED))
#
# Provides:
#   Numpy::Numpy       - imported interface target
#   NumPy_INCLUDE  - path to numpy headers
#   Numpy_FOUND        - true if found

if(NOT Python_EXECUTABLE)
  message(FATAL_ERROR "FindNumpy: Python_EXECUTABLE not set. Call find_package(Python) first.")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}"
          -c "import numpy as np; print(np.get_include())"
  OUTPUT_VARIABLE NumPy_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _numpy_result
)

if(_numpy_result EQUAL 0 AND EXISTS "${NumPy_INCLUDE}")
  set(Numpy_FOUND TRUE)
else()
  set(Numpy_FOUND FALSE)
endif()

message(STATUS "${NumPy_INCLUDE}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Numpy
  REQUIRED_VARS NumPy_INCLUDE
)

if(Numpy_FOUND AND NOT TARGET Numpy::Numpy)
  add_library(Numpy::Numpy INTERFACE IMPORTED)
  set_target_properties(Numpy::Numpy PROPERTIES
    INTERFACE_INCLUDEECTORIES "${NumPy_INCLUDE}"
  )
endif()
