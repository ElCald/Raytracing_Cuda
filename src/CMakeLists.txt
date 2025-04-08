# Adding source files
set(SOURCE_FILES
    main.cpp
    ${CMAKE_SOURCE_DIR}/Geometry/geometry.cpp
    ${CMAKE_SOURCE_DIR}/GeometricsObjects/forms.cpp
    ${CMAKE_SOURCE_DIR}/Utils/camera.cpp
    ${CMAKE_SOURCE_DIR}/Utils/scene.cpp
    ${CMAKE_SOURCE_DIR}/Utils/light.cpp
)

# Adding some optimal flags
add_compile_options(-Ofast -Wall -Wextra -march=native -mtune=native)

# Adding executable
add_executable(main ${SOURCE_FILES})


# Adding include folders
include_directories(
    ${CMAKE_SOURCE_DIR}/Geometry
    ${CMAKE_SOURCE_DIR}/GeometricObjects
    ${CMAKE_SOURCE_DIR}/Utils
)
