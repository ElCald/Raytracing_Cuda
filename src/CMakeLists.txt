# CMakeLists.txt dans le répertoire 'src'

# Ajouter les fichiers sources
set(SOURCE_FILES
    main.cpp
    ${CMAKE_SOURCE_DIR}/Geometry/geometry.cpp
    ${CMAKE_SOURCE_DIR}/GeometricObjects/formes.cpp
    ${CMAKE_SOURCE_DIR}/Cameras/camera.cpp
)

# Ajouter l'exécutable
add_executable(main ${SOURCE_FILES})

# Ajouter les répertoires d'inclusion
include_directories(
    ${CMAKE_SOURCE_DIR}/Geometry
    ${CMAKE_SOURCE_DIR}/GeometricObjects
    ${CMAKE_SOURCE_DIR}/Cameras
)
