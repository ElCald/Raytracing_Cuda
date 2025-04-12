# Adding source files
set(SOURCE_FILES
    main.cu
    ${CMAKE_SOURCE_DIR}/Geometry/geometry.cu
    ${CMAKE_SOURCE_DIR}/GeometricsObjects/forms.cu
    ${CMAKE_SOURCE_DIR}/Utils/camera.cu
    ${CMAKE_SOURCE_DIR}/Utils/scene.cu
    ${CMAKE_SOURCE_DIR}/Utils/light.cu
)

# Flags for C++ (host)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall -Wextra -march=native -mtune=native")

# Flags for CUDA (device)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler=-Wall,-Wextra")

# Adding executable
add_executable(main ${SOURCE_FILES})


# Adding include folders
include_directories(
    ${CMAKE_SOURCE_DIR}/Geometry
    ${CMAKE_SOURCE_DIR}/GeometricObjects
    ${CMAKE_SOURCE_DIR}/Utils
)
