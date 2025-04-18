################
#Unitary tests
###############
enable_testing()

include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
        DOWNLOAD_EXTRACT_TIMESTAMP true
)

# Un-comment if issues with GoogleTest
#FetchContent_MakeAvailable(googletest)

# Adding source files
set(SOURCE_FILES
    tests.cu
    ${CMAKE_SOURCE_DIR}/Geometry/geometry.cu
)

# Flags for C++ (host)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall -Wextra -march=native -mtune=native")

# Flags for CUDA (device)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler=-Wall,-Wextra -rdc=true")

# Creation of test executable
add_executable(tests ${SOURCE_FILES})

set_target_properties(tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link with GoogleTest
target_link_libraries(tests PRIVATE gtest gtest_main)

# Adding the test(s) in CMake
add_test(NAME ray_tracing_tests COMMAND tests)