# FindNumpy.cmake
# Finds NumPy include directory via the active Python interpreter.
#
# Requires Python_EXECUTABLE to be set (e.g. from find_package(Python REQUIRED))
#
# Provides:
#   Numpy::Numpy       - imported interface target
#   Numpy_INCLUDE_DIR  - path to numpy headers
#   Numpy_FOUND        - true if found

if(NOT Python_EXECUTABLE)
  message(FATAL_ERROR "FindNumpy: Python_EXECUTABLE not set. Call find_package(Python) first.")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}"
          -c "import numpy as np; print(np.get_include())"
  OUTPUT_VARIABLE Numpy_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _numpy_result
)

if(_numpy_result EQUAL 0 AND EXISTS "${Numpy_INCLUDE_DIR}")
  set(Numpy_FOUND TRUE)
else()
  set(Numpy_FOUND FALSE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Numpy
  REQUIRED_VARS Numpy_INCLUDE_DIR
)

if(Numpy_FOUND AND NOT TARGET Numpy::Numpy)
  add_library(Numpy::Numpy INTERFACE IMPORTED)
  set_target_properties(Numpy::Numpy PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Numpy_INCLUDE_DIR}"
  )
endif()
